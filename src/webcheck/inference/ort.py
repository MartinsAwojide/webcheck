"""The models without a Jetson: the Edge Impulse Studio ONNX exports run through ONNX Runtime behind
the call shape of the TensorRT models in `trt`, so the pipeline does not change. The gate's input is the CPU Lanczos
resize the pipeline falls back to; the GPU resize kernel stays with the TensorRT path. 
The GPU is reached directly (the vendor's CUDA or DirectML provider) or through the WebGPU plugin provider (Vulkan on
Linux, Metal on macOS), as the user picks with `--gpu`. Imports onnxruntime and the plugin lazily (the optional extras
`onnx` and `webgpu`), so the package imports without them."""
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from webcheck.inference.engine import ModelNotFoundError

PROVIDER_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider")
CPU = "CPUExecutionProvider"
WEBGPU = "WebGpuExecutionProvider"
ACCELERATOR = {"CUDAExecutionProvider": "GPU", "DmlExecutionProvider": "GPU", WEBGPU: "GPU", CPU: "CPU"}
PRECISION = {"CUDAExecutionProvider": "float32 with TF32 allowed"}      # the CUDA provider's use_tf32 defaults to 1
ROUTE_NAME = {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML", WEBGPU: "WebGPU"}
BACKENDS = ("tensorrt", "onnxruntime")
GPU_ROUTES = ("auto", "direct", "webgpu", "none")
_registered = set()


def choose_providers(available: Sequence[str]) -> list:
    """CUDA, then DirectML, from those available, with the CPU always last as the fallback."""
    return [p for p in PROVIDER_ORDER if p in available] + [CPU]


def backend_info(ort, provider: str) -> Tuple[str, str, str, str]:
    """(runtime, version, accelerator, precision) for `hmi.runtime_panel`; a GPU route is named after the version."""
    version = ort.__version__ + (f" · {ROUTE_NAME[provider]}" if provider in ROUTE_NAME else "")
    return ("ONNX Runtime", version, ACCELERATOR.get(provider, "CPU"), PRECISION.get(provider, "float32"))


def choose_route(gpu: str, available: Sequence[str], webgpu_ready: bool) -> str:
    """direct, webgpu or cpu for the user's --gpu: auto takes direct, then WebGPU, then the CPU."""
    if gpu not in GPU_ROUTES:
        raise ValueError(f"--gpu must be one of {', '.join(GPU_ROUTES)}, got {gpu!r}")
    direct = any(p in available for p in PROVIDER_ORDER)
    if gpu == "none":
        return "cpu"
    if gpu == "direct" and not direct:
        raise ValueError("no direct GPU provider here: install onnxruntime-gpu (NVIDIA, CUDA) or onnxruntime-directml "
                         "(Windows) in place of onnxruntime, or use --gpu webgpu")
    if gpu == "webgpu" and not webgpu_ready:
        raise ValueError("WebGPU is not available: uv sync --extra webgpu (Python 3.11+, macOS on Apple silicon); on "
                         "Linux the Vulkan loader libvulkan.so.1 must be installed")
    if gpu in ("direct", "webgpu"):
        return gpu
    return "direct" if direct else "webgpu" if webgpu_ready else "cpu"


def _webgpu_plugin():
    try:
        import onnxruntime_ep_webgpu
    except ImportError:
        return None
    return onnxruntime_ep_webgpu


def webgpu_device(ort, plugin=None):
    """The WebGPU plugin's device, registering the plugin with `ort` once; None without the plugin or an adapter."""
    plugin = plugin or _webgpu_plugin()
    if plugin is None:
        return None
    if id(ort) not in _registered:
        try:
            ort.register_execution_provider_library("webgpu", plugin.get_library_path())
        except Exception:
            return None
        _registered.add(id(ort))
    name = plugin.get_ep_name()
    return next((d for d in ort.get_ep_devices() if d.ep_name == name), None)


def _ort():
    import onnxruntime
    return onnxruntime


def load_session(path, ort=None, gpu: str = "auto", webgpu_ep=None):
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"no model at {path}")
    ort = ort or _ort()
    available = ort.get_available_providers()
    device = None
    if gpu == "webgpu" or (gpu == "auto" and not any(p in available for p in PROVIDER_ORDER)):
        device = webgpu_device(ort, webgpu_ep)
    route = choose_route(gpu, available, device is not None)
    if route == "webgpu":
        options = ort.SessionOptions()
        options.add_provider_for_devices([device], {})
        return ort.InferenceSession(str(path), sess_options=options)
    return ort.InferenceSession(str(path), providers=choose_providers(available) if route == "direct" else [CPU])


def _same_shape(a, b) -> bool:
    """Shapes match where both dimensions are fixed; a named or unknown dimension matches anything."""
    if len(a) != len(b):
        return False
    return all(not isinstance(x, int) or not isinstance(y, int) or x == y for x, y in zip(a, b))


class _OrtModel:
    """One session with one input and one output, float32, batch 1."""

    def __init__(self, session):
        self._session = session
        self._in, self._out = session.get_inputs()[0], session.get_outputs()[0]
        self.input_shape, self.output_shape = list(self._in.shape), list(self._out.shape)

    def __call__(self, x) -> np.ndarray:
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]
                or not _same_shape(self.input_shape, x.shape)):
            shape = getattr(x, "shape", None)
            raise ValueError(f"input must be C-contiguous float32 {self.input_shape}, got {shape} {getattr(x, 'dtype', None)}")
        return np.asarray(self._session.run([self._out.name], {self._in.name: x})[0], dtype=np.float32)


class OrtGateModel:
    """The FOMO-AD gate: backbone (1x320x320x3 -> 1x40x40x120) then head (-> 1x40x40 anomaly map)."""

    def __init__(self, backbone_session, head_session):
        self._backbone, self._head = _OrtModel(backbone_session), _OrtModel(head_session)
        if not _same_shape(self._backbone.output_shape, self._head.input_shape):
            raise ValueError(f"backbone output {self._backbone.output_shape} does not feed head input {self._head.input_shape}")

    def __call__(self, gate_input) -> np.ndarray:
        return self._head(np.ascontiguousarray(self._backbone(gate_input)))


class OrtDetectorModel:
    """YOLO-Pro: 1x640x640x3 -> 1x8400x10 (decode with webcheck.inference.detector.decode_detections)."""

    def __init__(self, session):
        self._model = _OrtModel(session)

    def __call__(self, detector_input: np.ndarray) -> np.ndarray:
        return self._model(detector_input)


def load_models(gate_dir, det_dir, ort=None, gpu: str = "auto", webgpu_ep=None):
    """(gate, detector, provider) from the demo folders on the `gpu` route, as the TensorRT path loads them."""
    ort = ort or _ort()
    backbone = load_session(Path(gate_dir) / "trained.onnx", ort, gpu, webgpu_ep)
    on_webgpu = backbone.get_providers()[0] == WEBGPU                  # the head (MatMul, TopK) is faster on the CPU there
    head = load_session(Path(gate_dir) / "trained_head.onnx", ort, "none" if on_webgpu else gpu, webgpu_ep)
    det = load_session(Path(det_dir) / "trained.onnx", ort, gpu, webgpu_ep)
    return OrtGateModel(backbone, head), OrtDetectorModel(det), det.get_providers()[0]


def select_backend(requested, tensorrt_available: bool, onnxruntime_available: bool) -> str:
    """TensorRT where it imports, ONNX Runtime otherwise, unless one is asked for."""
    if requested is not None and requested not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {requested!r}")
    if requested == "tensorrt" and not tensorrt_available:
        raise ValueError("TensorRT is not installed here (it comes with JetPack on the Jetson); use --backend onnxruntime")
    if requested == "onnxruntime" and not onnxruntime_available:
        raise ValueError("ONNX Runtime is not installed: uv sync --extra onnx (or pip install onnxruntime)")
    if requested is not None:
        return requested
    if tensorrt_available:
        return "tensorrt"
    if onnxruntime_available:
        return "onnxruntime"
    raise ValueError("neither TensorRT nor ONNX Runtime is installed: uv sync --extra onnx (or pip install onnxruntime)")
