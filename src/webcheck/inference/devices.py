"""The device options the splash offers: what can run the models on this machine, best first, each
with the speed the line starts at. The speeds are fixed here from measurements made beforehand, never timed on the
user's machine, and capped at the demo's starting speed."""
from dataclasses import dataclass
from typing import Optional, Sequence

from webcheck.inference.hmi import DEFAULT_SPEED_M_PER_MIN

DEFAULT_SPEEDS = {
    "tensorrt": DEFAULT_SPEED_M_PER_MIN,     # the Jetson keeps up far above the cap
    "direct": 8.0,                           # not measured: WebGPU's figure, as a floor
    "webgpu": 8.0,                           # the self-check's ladder rule (selfcheck.ladder_result)
    "cpu": 1.0,                              # the slowest speed the line runs at
}
_DIRECT_NAMES = {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML"}


@dataclass(frozen=True)
class DeviceOption:
    key: str
    label: str
    backend: str
    gpu: Optional[str]              # ONNX Runtime's route (ort.GPU_ROUTES), None for TensorRT
    speed: float

    def as_choice(self) -> dict:
        """What the splash shows for it."""
        return {"key": self.key, "label": self.label, "speed": self.speed}


def detect_options(tensorrt_available: bool, onnxruntime_available: bool, providers: Sequence[str],
                   webgpu_ready: bool) -> list:
    """TensorRT, then ONNX Runtime's direct GPU provider, WebGPU and the CPU, as far as they are here."""
    options = []
    if tensorrt_available:
        options.append(DeviceOption("tensorrt", "TensorRT · GPU", "tensorrt", None, DEFAULT_SPEEDS["tensorrt"]))
    if onnxruntime_available:
        direct = next((name for p, name in _DIRECT_NAMES.items() if p in providers), None)
        if direct:
            options.append(DeviceOption("direct", f"ONNX Runtime · {direct}", "onnxruntime", "direct", DEFAULT_SPEEDS["direct"]))
        if webgpu_ready:
            options.append(DeviceOption("webgpu", "ONNX Runtime · WebGPU", "onnxruntime", "webgpu", DEFAULT_SPEEDS["webgpu"]))
        options.append(DeviceOption("cpu", "ONNX Runtime · CPU", "onnxruntime", "none", DEFAULT_SPEEDS["cpu"]))
    return options


def preferred_option(options: Sequence[DeviceOption], backend: Optional[str], gpu: Optional[str]) -> DeviceOption:
    """The option the splash picks first: the one the driver's --backend/--gpu name where it is here, else the best."""
    def wanted(o):
        if backend == "tensorrt":
            return o.key == "tensorrt"
        if backend == "onnxruntime" and gpu in (None, "auto"):
            return o.backend == "onnxruntime"
        if gpu not in (None, "auto"):
            return o.gpu == gpu
        return True
    return next((o for o in options if wanted(o)), options[0])
