"""TensorRT glue for the Jetson Orin Nano (host TensorRT 10.3, CUDA 12 runtime). Thin on purpose: all the
logic that can be tested without a GPU lives in the sibling modules; this file only builds engines and runs
them, and is checked on the board against Edge Impulse Studio's scores.

Imports tensorrt lazily so the rest of the package imports (and is tested) on a machine without it."""
import ctypes
from typing import Tuple

import numpy as np

MEMCPY_H2D, MEMCPY_D2H = 1, 2


def tensorrt_version() -> str:
    import tensorrt as trt
    return trt.__version__


def build_engine(onnx_bytes: bytes) -> bytes:
    """Build an fp32 engine from ONNX bytes (the board takes a few minutes for the 29 MB detector)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        raise RuntimeError(str(parser.get_error(0)))
    serialized = builder.build_serialized_network(network, builder.create_builder_config())
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine")
    return bytes(serialized)


class Cuda:
    """The few CUDA runtime calls needed, through ctypes (no pycuda dependency)."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(self.stream)))

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f"CUDA error {code}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr

    def copy(self, dst, src, nbytes: int, kind: int):
        self.check(self.lib.cudaMemcpyAsync(dst, src, ctypes.c_size_t(nbytes), kind, self.stream))

    def sync(self):
        self.check(self.lib.cudaStreamSynchronize(self.stream))


_cuda = None


def get_cuda() -> Cuda:
    global _cuda
    if _cuda is None:
        _cuda = Cuda()
    return _cuda


class DeviceInput:
    """A float32 model input that is already on the GPU, given by the address of its first element (the CUDA
    resize writes the six gate inputs into one buffer on the same stream)."""

    def __init__(self, address: int):
        self.address = int(address)


class TrtModel:
    """One engine with one input and one output tensor, float32, batch 1."""

    def __init__(self, engine_bytes: bytes):
        import tensorrt as trt
        self._cuda = get_cuda()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("could not deserialize the engine (built by another TensorRT version?)")
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        modes = {n: self._engine.get_tensor_mode(n) for n in names}
        self._in = next(n for n in names if modes[n] == trt.TensorIOMode.INPUT)
        self._out = next(n for n in names if modes[n] == trt.TensorIOMode.OUTPUT)
        self.input_shape: Tuple[int, ...] = tuple(self._engine.get_tensor_shape(self._in))
        self.output_shape: Tuple[int, ...] = tuple(self._engine.get_tensor_shape(self._out))
        self._ctx = self._engine.create_execution_context()
        self._d_in = self._cuda.malloc(int(np.prod(self.input_shape)) * 4)
        self._d_out = self._cuda.malloc(int(np.prod(self.output_shape)) * 4)
        self._bound_in = self._d_in.value
        self._ctx.set_tensor_address(self._in, self._bound_in)
        self._ctx.set_tensor_address(self._out, self._d_out.value)
        self._host_out = np.empty(self.output_shape, np.float32)

    @property
    def output_address(self) -> int:
        return self._d_out.value

    def _bind_input(self, address: int) -> None:
        if address != self._bound_in:
            self._ctx.set_tensor_address(self._in, address)
            self._bound_in = address

    def enqueue(self, x) -> None:
        """Queue one run on the shared stream. `x` is a host array (copied to the GPU first) or a DeviceInput
        (already on the GPU, read in place)."""
        cu = self._cuda
        if isinstance(x, DeviceInput):
            self._bind_input(x.address)
        else:
            if x.shape != self.input_shape or x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
                raise ValueError(f"input must be C-contiguous float32 {self.input_shape}, got {x.shape} {x.dtype}")
            self._bind_input(self._d_in.value)
            cu.copy(self._d_in, x.ctypes.data_as(ctypes.c_void_p), x.nbytes, MEMCPY_H2D)
        self._ctx.execute_async_v3(cu.stream.value)

    def read(self) -> np.ndarray:
        """Wait for the queued run and return its output."""
        cu = self._cuda
        cu.copy(self._host_out.ctypes.data_as(ctypes.c_void_p), self._d_out, self._host_out.nbytes, MEMCPY_D2H)
        cu.sync()
        return self._host_out.copy()

    def __call__(self, x) -> np.ndarray:
        self.enqueue(x)
        return self.read()


class GateModel:
    """The FOMO-AD gate: backbone (1x320x320x3 -> 1x40x40x120) then head (-> 1x40x40 anomaly map)."""

    def __init__(self, backbone_engine: bytes, head_engine: bytes):
        self._backbone, self._head = TrtModel(backbone_engine), TrtModel(head_engine)
        if self._backbone.output_shape != self._head.input_shape:
            raise ValueError(f"backbone output {self._backbone.output_shape} does not feed head input {self._head.input_shape}")

    def __call__(self, gate_input) -> np.ndarray:
        """The head reads the backbone's output where it is on the GPU, so the pair needs one sync, not two."""
        self._backbone.enqueue(gate_input)
        self._head.enqueue(DeviceInput(self._backbone.output_address))
        return self._head.read()


class DetectorModel:
    """YOLO-Pro: 1x640x640x3 -> 1x8400x10 (decode with webcheck.inference.detector.decode_detections)."""

    def __init__(self, engine: bytes):
        self._model = TrtModel(engine)

    def __call__(self, detector_input: np.ndarray) -> np.ndarray:
        return self._model(detector_input)
