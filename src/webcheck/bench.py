"""The benchmark rig. Every figure carries the full configuration tuple, and the labels that matter
are read rather than typed: the rung and board from the host (device-tree model and /etc/nv_tegra_release on a Jetson,
a container marker for R2, otherwise the PC's CPU for R1), the numeric format from TensorRT's default builder flags (how
the project builds its engines, `inference.trt.build_engine`). A measured result can only come from `run_benchmark`,
which checks the board is cool and idle first, discards warm-up blocks, times at least five measured blocks and keeps the
temperatures and load around each one; a figure from anywhere else is an `Estimate` that names its source."""
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

MIN_REPETITIONS = 5
ORIN_NANO = "Jetson Orin Nano"
FORMATS = ("float32", "float32 with TF32 allowed")
_RESULT_KEY = object()                  # only run_benchmark holds it, so a measured result cannot be made by hand


class NotReady(RuntimeError):
    """The host cannot be measured as it is: unknown board, reduced precision, too hot or too loaded."""


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError, TypeError):    # TypeError: a sysfs file answering EAGAIN (Orin Nano cv* zones)
        return None


@dataclass(frozen=True)
class Host:
    rung: str
    board: str
    l4t: Optional[str]


def detect_host(root="/") -> Host:
    """The rung and board of the machine under `root` (the real one by default)."""
    root = Path(root)
    model = _read(root / "proc/device-tree/model")
    if model is not None and model.strip("\0\n "):
        model = model.strip("\0\n ")
        if ORIN_NANO not in model:
            raise NotReady(f"this rig measures the Jetson Orin Nano (R4), found {model!r}")
        release = _read(root / "etc/nv_tegra_release") or ""
        l4t = None
        if release.startswith("# R"):
            major = release[2:].split(" ", 1)[0]
            revision = release.split("REVISION:", 1)[1].split(",", 1)[0].strip() if "REVISION:" in release else ""
            l4t = f"{major}.{revision}" if revision else major
        return Host("R4", model, l4t)
    cpu = next((line.split(":", 1)[1].strip() for line in (_read(root / "proc/cpuinfo") or "").splitlines()
                if line.startswith("model name")), None)
    if not cpu:
        raise NotReady("the host has neither a device-tree model nor a CPU model name")
    return Host("R2" if (root / ".dockerenv").exists() else "R1", cpu, None)


def read_temperatures(root="/") -> Dict[str, float]:
    """Degrees Celsius per thermal zone; zones that cannot be read right now are left out."""
    temps = {}
    for zone in sorted(Path(root, "sys/class/thermal").glob("thermal_zone*")):
        kind, value = _read(zone / "type"), _read(zone / "temp")
        try:
            temps[kind.strip()] = int(value.strip()) / 1000.0
        except (AttributeError, ValueError):
            continue
    return temps


def read_load(root="/") -> float:
    """The one-minute load average."""
    return float((_read(Path(root, "proc/loadavg")) or "").split()[0])


def _run(argv):
    done = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    return done.returncode, done.stdout


def power_mode(run: Callable = _run) -> Optional[str]:
    """The Jetson power mode from `nvpmodel -q` (no sudo needed), or None where it cannot be read."""
    try:
        code, out = run(["nvpmodel", "-q"])
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    if code != 0 or not first.startswith("NV Power Mode:"):
        return None
    return first.split(":", 1)[1].strip() or None


def numeric_format(trt) -> str:
    """The format engines get from a default builder config, as `inference.trt.build_engine` builds them."""
    config = trt.Builder(trt.Logger(trt.Logger.WARNING)).create_builder_config()
    if config.get_flag(trt.BuilderFlag.FP16) or config.get_flag(trt.BuilderFlag.INT8):
        raise NotReady("reduced precision (FP16/INT8) is out of this project's scope")
    return "float32 with TF32 allowed" if config.get_flag(trt.BuilderFlag.TF32) else "float32"


def check_ready(temperatures: Dict[str, float], load: float, *, max_temp_c: float, max_load: float) -> Dict:
    """Refuse a hot or loaded host; the limits are the caller's, inclusive. Returns the readings."""
    if not temperatures:
        raise NotReady("no temperature could be read, so the board's state is unknown")
    hot = {zone: t for zone, t in temperatures.items() if t > max_temp_c}
    if hot:
        raise NotReady(f"too hot to measure (limit {max_temp_c} C): {hot}")
    if load > max_load:
        raise NotReady(f"too loaded to measure: load {load} over {max_load}")
    return {"temperatures_c": dict(temperatures), "load_1min": load}


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Configuration:
    """The full configuration tuple: models (by sha256), numeric format, runtime, host (rung, board, L4T), input shape,
    power mode, versions and anything else that changes the number (`extras`, e.g. MQTT on)."""
    host: Host
    numeric_format: str
    runtime: str
    models: Dict[str, Dict[str, str]]
    input_shape: Dict
    power_mode: Optional[str]
    versions: Dict[str, str]
    extras: Dict = field(default_factory=dict)

    @classmethod
    def build(cls, *, host: Host, numeric_format: str, runtime: str, models: Dict[str, str], input_shape: Dict,
              power_mode: Optional[str], versions: Dict[str, str], extras: Optional[Dict] = None) -> "Configuration":
        if not isinstance(host, Host):
            raise ValueError("host must come from detect_host")
        if numeric_format not in FORMATS:
            raise ValueError(f"numeric_format must be one of {FORMATS} (from numeric_format()), got {numeric_format!r}")
        if not isinstance(runtime, str) or not runtime:
            raise ValueError("runtime must name the runtime and its version")
        if not isinstance(models, dict) or not models:
            raise ValueError("models must name at least one model file")
        if not isinstance(input_shape, dict) or not input_shape:
            raise ValueError("input_shape must describe the input")
        hashed = {}
        for name, path in models.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"model {name!r}: no file at {path}")
            hashed[name] = {"path": str(path), "sha256": sha256_of(path)}
        return cls(host, numeric_format, runtime, hashed, dict(input_shape), power_mode, dict(versions), dict(extras or {}))

    def to_dict(self) -> Dict:
        return {"rung": self.host.rung, "board": self.host.board, "l4t": self.host.l4t,
                "numeric_format": self.numeric_format, "runtime": self.runtime, "models": self.models,
                "input_shape": self.input_shape, "power_mode": self.power_mode, "versions": self.versions,
                "extras": self.extras}


def _spread(values: Sequence[float]) -> Dict[str, float]:
    return {"median": statistics.median(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


class BenchmarkResult:
    """A measured result: only `run_benchmark` can make one."""
    kind = "measured"

    def __init__(self, *, name: str, configuration: Configuration, seconds: List[float], items: List[int],
                 readings: List[Dict], warmups: int = 1, _key=None):
        if _key is not _RESULT_KEY:
            raise TypeError("a measured result comes only from run_benchmark")
        self.name, self.configuration, self.warmups = name, configuration, warmups
        self.seconds, self.items, self.readings = list(seconds), list(items), list(readings)
        self.ms_per_item = _spread([1000.0 * s / n for s, n in zip(self.seconds, self.items)])
        self.items_per_s = _spread([n / s for s, n in zip(self.seconds, self.items)])

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "name": self.name, "configuration": self.configuration.to_dict(),
                "warmups": self.warmups, "repetitions": len(self.seconds), "seconds": self.seconds, "items": self.items,
                "ms_per_item": self.ms_per_item, "items_per_s": self.items_per_s, "readings": self.readings}


def wait_until_ready(check: Callable[[], Dict], *, timeout_s: float, poll_s: float = 5.0,
                     sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> Dict:
    """Poll `check` (it raises NotReady) until it passes, e.g. while the load average settles after loading engines;
    after `timeout_s` the last NotReady is raised."""
    start = clock()
    while True:
        try:
            return check()
        except NotReady:
            if clock() - start >= timeout_s:
                raise
        sleep(poll_s)


def run_benchmark(block: Callable[[], int], *, name: str, configuration: Configuration, readiness: Callable[[], Dict],
                  repetitions: int = MIN_REPETITIONS, warmups: int = 1, clock: Callable[[], float] = time.perf_counter,
                  reader: Optional[Callable[[], Dict]] = None) -> BenchmarkResult:
    """Time `block` (which does a fixed amount of work and returns how many items it did, e.g. slices): `readiness()`
    must pass first (it raises NotReady otherwise), then `warmups` blocks are run and dropped, then `repetitions`
    blocks are timed, with `reader()` (default: `readiness`) read before and after each. The run's own work raises the
    load, so a reader that only reads, not checks, keeps the run going once it has started."""
    reader = readiness if reader is None else reader
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < MIN_REPETITIONS:
        raise ValueError(f"at least {MIN_REPETITIONS} measured repetitions, got {repetitions!r}")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 1:
        raise ValueError(f"at least one warm-up block, got {warmups!r}")
    if not isinstance(configuration, Configuration):
        raise ValueError("configuration must be a Configuration")
    readiness()
    for _ in range(warmups):
        _items(block())
    seconds, items, readings = [], [], []
    for _ in range(repetitions):
        before = reader()
        start = clock()
        done = _items(block())
        seconds.append(clock() - start)
        items.append(done)
        readings.append({"before": before, "after": reader()})
    return BenchmarkResult(name=name, configuration=configuration, seconds=seconds, items=items, readings=readings,
                           warmups=warmups, _key=_RESULT_KEY)


def _items(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"a block must return how many items it did (a positive whole number), got {value!r}")
    return value


class Estimate:
    """A figure not measured here (an Edge Impulse table, a datasheet): it names its source and when it was read."""
    kind = "estimate"

    def __init__(self, *, name: str, value: float, unit: str, source: str, read_on: str, configuration: Configuration):
        for label, text in (("name", name), ("unit", unit), ("source", source), ("read_on", read_on)):
            if not isinstance(text, str) or not text:
                raise ValueError(f"an estimate needs its {label}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"an estimate's value must be a finite number, got {value!r}")
        if not isinstance(configuration, Configuration):
            raise ValueError("configuration must be a Configuration")
        self.name, self.value, self.unit, self.source, self.read_on = name, float(value), unit, source, read_on
        self.configuration = configuration

    def to_dict(self) -> Dict:
        return {"kind": self.kind, "name": self.name, "value": self.value, "unit": self.unit, "source": self.source,
                "read_on": self.read_on, "configuration": self.configuration.to_dict()}


def write_report(entries: Sequence, path) -> Path:
    """One JSON file with each result or estimate and its configuration tuple."""
    if not all(isinstance(e, (BenchmarkResult, Estimate)) for e in entries):
        raise ValueError("a report holds only BenchmarkResult and Estimate entries")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": [e.to_dict() for e in entries]}, indent=1, allow_nan=False) + "\n")
    return path
