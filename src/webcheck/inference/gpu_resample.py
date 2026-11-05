"""The gate's tile resize on the GPU: every tile's 640 -> 320 Lanczos resize in two kernel launches (resample.cu), bytes
equal to Pillow's. Thin glue like trt.py: the logic that can be tested without a GPU is in resample.py, and this file is
checked on the board against Pillow.

Needs nvcc on the board (CUDA 12.6 toolkit); the library is built once and cached by the source's hash."""
import ctypes
import hashlib
import os
import shutil
import subprocess

import numpy as np

from webcheck.inference.frame import FRAME_HEIGHT, FRAME_WIDTH, TILE_SIZE, FrameError, tile_origins
from webcheck.inference.preprocess import GATE_SIZE
from webcheck.inference.resample import make_plan
from webcheck.inference.trt import MEMCPY_D2H, MEMCPY_H2D, DeviceInput, get_cuda

CU_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resample.cu")
_NVCC_FALLBACK = "/usr/local/cuda/bin/nvcc"
_HOST_ALLOC_MAPPED = 2          # cudaHostAllocMapped: host memory the GPU reads in place (the Orin shares its RAM)
_FRAME_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3


def build_library(cache_dir: str) -> str:
    """Compile resample.cu with nvcc into cache_dir (once per source version) and return the .so path."""
    with open(CU_SOURCE, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"libresample-{digest}.so")
    if os.path.exists(path):
        return path
    nvcc = shutil.which("nvcc") or _NVCC_FALLBACK
    tmp = path + ".tmp"
    cmd = [nvcc, "-O3", "-arch=native", "--cudart", "shared", "-shared", "-Xcompiler", "-fPIC", CU_SOURCE, "-o", tmp]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"nvcc failed: {done.stderr.strip()}")
    os.replace(tmp, path)
    return path


class GpuTileResizer:
    """Resizes the 640x640 tiles of a 1920x1080 RGB frame (all six, or those of the given tile columns) to 320x320 on
    the GPU.

    The frame lives in mapped host memory that the kernels read in place, so a frame that a capture wrote into
    `frame_buffer` costs no upload. Any other frame is copied into that buffer first. This relies on the Jetson's
    shared memory; on a discrete GPU the kernels would read across PCIe."""

    def __init__(self, cache_dir: str, columns=None):
        self._cu = get_cuda()
        self._lib = ctypes.CDLL(build_library(cache_dir))
        pv, ci = ctypes.c_void_p, ctypes.c_int
        self._lib.webcheck_resample_tiles.argtypes = [pv, ci, pv, ci, ci, ci, ci, ci, ci, ci, pv, pv, pv, pv, pv, pv, ci, pv]
        self._lib.webcheck_resample_tiles.restype = ci
        plan = make_plan(TILE_SIZE, TILE_SIZE, GATE_SIZE, GATE_SIZE)
        if not (plan.need_horizontal and plan.need_vertical):
            raise ValueError("the GPU path runs both passes; the tile and gate sizes must differ")
        self._plan = plan
        self._device_buffers = []
        self._tables = [self._upload(np.ascontiguousarray(a, dtype=np.int32)) for a in
                        (plan.horizontal_bounds, plan.horizontal_kk, plan.vertical_bounds, plan.vertical_kk)]
        origins = tile_origins(columns)
        self._n = len(origins)
        self._xy = np.array([[x, y] for _, _, x, y in origins], np.int32).ravel()
        n, w, runtime = self._n, GATE_SIZE, self._cu.lib
        self._host_frame = ctypes.c_void_p()
        self._cu.check(runtime.cudaHostAlloc(ctypes.byref(self._host_frame), ctypes.c_size_t(_FRAME_BYTES), _HOST_ALLOC_MAPPED))
        self._frame = ctypes.c_void_p()      # the GPU's address of the same memory
        self._cu.check(runtime.cudaHostGetDevicePointer(ctypes.byref(self._frame), self._host_frame, 0))
        self._frame_buf = np.ctypeslib.as_array((ctypes.c_uint8 * _FRAME_BYTES).from_address(self._host_frame.value)).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, 3)
        self._tmp = self._malloc(n * plan.rows * w * 3)
        self._out_u8 = self._malloc(n * w * w * 3)
        self._out_f32 = self._malloc(n * w * w * 3 * 4)
        self._host_u8 = np.empty((n, w, w, 3), np.uint8)
        self._host_f32 = np.empty((n, w, w, 3), np.float32)

    def _malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = self._cu.malloc(nbytes)
        self._device_buffers.append(ptr)
        return ptr

    def _upload(self, a: np.ndarray) -> ctypes.c_void_p:
        ptr = self._malloc(a.nbytes)
        self._cu.copy(ptr, a.ctypes.data_as(ctypes.c_void_p), a.nbytes, MEMCPY_H2D)
        self._cu.sync()
        return ptr

    @property
    def tiles(self) -> int:
        return self._n

    @property
    def frame_buffer(self) -> np.ndarray:
        """The 1080x1920x3 uint8 array the kernels read; a capture that writes frames here skips the copy."""
        return self._frame_buf

    def upload(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise FrameError(f"frame must be a {FRAME_HEIGHT}x{FRAME_WIDTH}x3 uint8 array")
        if frame.ctypes.data != self._frame_buf.ctypes.data or frame.strides != self._frame_buf.strides:
            np.copyto(self._frame_buf, frame)

    def launch(self, as_float: bool) -> None:
        """Queue both passes on the stream; the result is in the device buffer returned by `device_output`."""
        p = self._plan
        hb, hk, vb, vk = (t.value for t in self._tables)
        code = self._lib.webcheck_resample_tiles(
            self._frame.value, FRAME_WIDTH * 3, self._xy.ctypes.data_as(ctypes.c_void_p), self._n, p.first_row, p.rows,
            GATE_SIZE, GATE_SIZE, p.ksize_horizontal, p.ksize_vertical, hb, hk, vb, vk, self._tmp.value,
            self.device_output(as_float).value, int(as_float), self._cu.stream.value)
        if code != 0:
            raise RuntimeError(f"resample kernel failed with CUDA error {code}")

    def device_output(self, as_float: bool) -> ctypes.c_void_p:
        return self._out_f32 if as_float else self._out_u8

    def sync(self) -> None:
        self._cu.sync()

    def resize(self, frame: np.ndarray, as_float: bool = False) -> np.ndarray:
        """The tiles resized, as an (n, 320, 320, 3) uint8 array, or float32 divided by 255 with as_float."""
        self.upload(frame)
        self.launch(as_float)
        host = self._host_f32 if as_float else self._host_u8
        self._cu.copy(host.ctypes.data_as(ctypes.c_void_p), self.device_output(as_float), host.nbytes, MEMCPY_D2H)
        self._cu.sync()
        return host.copy()

    def close(self) -> None:
        """Free every buffer this object allocated."""
        if getattr(self, "_host_frame", None) is None or not self._host_frame.value:
            return
        runtime = self._cu.lib
        self._cu.sync()
        for ptr in self._device_buffers:
            runtime.cudaFree(ptr)
        self._device_buffers = []
        runtime.cudaFreeHost(self._host_frame)
        self._host_frame = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GpuGateInputs:
    """`process_frame(..., gate_inputs=GpuGateInputs(cache_dir, columns))`: the gate inputs of the tiles in `columns`
    (default all) are produced on the GPU and handed to the gate where they are, in the stream the gate runs in, so
    nothing goes back to the host. Use the same `columns` as the `process_frame` call."""

    def __init__(self, cache_dir: str, columns=None):
        self._resizer = GpuTileResizer(cache_dir, columns)
        self._tile_bytes = GATE_SIZE * GATE_SIZE * 3 * 4

    @property
    def frame_buffer(self) -> np.ndarray:
        return self._resizer.frame_buffer

    def __call__(self, frame: np.ndarray):
        self._resizer.upload(frame)
        self._resizer.launch(as_float=True)
        base = self._resizer.device_output(as_float=True).value
        return [DeviceInput(base + i * self._tile_bytes) for i in range(self._resizer.tiles)]
