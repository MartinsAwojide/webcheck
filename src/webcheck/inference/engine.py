"""TensorRT engine cache and the call guard. An engine is built once from an ONNX graph and reused until the
graph bytes or the TensorRT version change (engines do not carry across TensorRT versions)."""
import hashlib
import math
import os
import threading
from pathlib import Path
from typing import Callable


class ModelNotFoundError(FileNotFoundError):
    """The ONNX file to build an engine from is not there."""


class EngineBuildError(RuntimeError):
    """TensorRT could not build an engine from the graph."""


class InferenceTimeout(TimeoutError):
    """A model call did not finish in time."""


class EngineCache:
    def __init__(self, cache_dir, tensorrt_version: str, build: Callable[[bytes], bytes]):
        self._dir = Path(cache_dir)
        self._version = tensorrt_version
        self._build = build

    def get(self, onnx_path) -> bytes:
        path = Path(onnx_path)
        if not path.is_file():
            raise ModelNotFoundError(f"model file not found: {path}")
        onnx = path.read_bytes()
        key = hashlib.sha256(onnx + b"|" + self._version.encode()).hexdigest()[:32]
        cached = self._dir / f"{path.stem}-{key}.engine"
        if cached.is_file() and cached.stat().st_size > 0:
            return cached.read_bytes()
        try:
            engine = self._build(onnx)
        except Exception as exc:
            raise EngineBuildError(f"could not build an engine from {path.name}: {exc}") from exc
        if not engine:
            raise EngineBuildError(f"building an engine from {path.name} returned nothing")
        self._dir.mkdir(parents=True, exist_ok=True)
        part = cached.with_suffix(".part")
        part.write_bytes(engine)
        os.replace(part, cached)
        return engine


def run_with_timeout(fn, timeout_s: float, *args):
    """Call fn(*args); raise InferenceTimeout if it takes longer than timeout_s seconds. The call runs on a
    daemon thread, so a hung GPU call cannot keep the process from exiting; errors inside fn pass through."""
    if not (timeout_s > 0) or math.isinf(timeout_s):
        raise ValueError(f"timeout must be a positive finite number of seconds, got {timeout_s!r}")
    box = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise InferenceTimeout(f"call did not finish within {timeout_s} s")
    if "error" in box:
        raise box["error"]
    return box["value"]
