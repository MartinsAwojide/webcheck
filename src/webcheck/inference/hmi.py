"""The demo HMI's line: a belt the operator drives from the page.

`web_order_pairs` turns a placement's manifest into the roll in machine-direction order, left and right paired at each
position. `LineControl` holds what the page sets (speed, pause, step, the camera shown) and keeps a
timeline of it, so the belt's travel between two looks is exact even if the operator changed something in between;
`check_command` is the one place a command from the page is checked. `ControlledSource` is a driver source (driver.py)
whose belt follows that timeline and stands still while paused, so a pause never shows up as backlog; it
stamps each slice with the moment the belt reached it, and `TimedGeometry` logs those moments, so a speed change moves
the times and never the positions. `ReadAhead` loads the next frames in a thread, so the roll need not fit in memory."""
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from webcheck.datagen.scene import frame_geometry
from webcheck.inference.defect_log import LineGeometry
from webcheck.inference.driver import Arrival
from webcheck.inference.overlap import CAMERAS

SPEED_MIN_M_PER_MIN, SPEED_MAX_M_PER_MIN = 1.0, 450.0
DEFAULT_SPEED_M_PER_MIN = 123.0   # the demo's starting speed
_START_FIELDS = ("placement", "speed", "view")
POLL_S = 0.05                     # how soon a stopped belt asks to be looked at again
_PITCH_TOLERANCE_MM = 1e-6
INSPECTION_TITLES = {"A": "Post-print inspection", "B": "Post-lamination inspection"}   # A: after the press; B: after lamination (sees both)
_UNKNOWN_INSPECTION = "Defect inspection"


@dataclass(frozen=True)
class GateChoice:
    """A named anomaly gate the operator can pick on the splash screen: its ONNX folder and its threshold."""
    folder: str
    threshold: float


def parse_gates(specs) -> Dict[str, GateChoice]:
    """`--gate NAME=DIR:THRESHOLD` values, in the order given; the threshold is after the last colon."""
    gates: Dict[str, GateChoice] = {}
    for spec in specs:
        name, sep, rest = spec.partition("=")
        folder, colon, value = rest.rpartition(":")
        if not sep or not colon or not name or not folder or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError(f"a gate is NAME=DIR:THRESHOLD with NAME of letters, digits, _ or -, got {spec!r}")
        try:
            threshold = float(value)
        except ValueError:
            raise ValueError(f"gate {name!r}: the threshold must be a number, got {value!r}") from None
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"gate {name!r}: the threshold must be a positive finite number, got {value!r}")
        if name in gates:
            raise ValueError(f"gate {name!r} is named twice")
        gates[name] = GateChoice(folder, threshold)
    return gates


def inspection_title(placement: str) -> str:
    """What the HMI calls a station's inspection: print defects at placement A, lamination defects at B."""
    return INSPECTION_TITLES.get(placement, _UNKNOWN_INSPECTION)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_speed(speed) -> float:
    if not _is_number(speed) or not SPEED_MIN_M_PER_MIN <= speed <= SPEED_MAX_M_PER_MIN:
        raise ValueError(f"speed must be a number from {SPEED_MIN_M_PER_MIN:g} to {SPEED_MAX_M_PER_MIN:g} m/min, got {speed!r}")
    return float(speed)


def _check_view(view) -> str:
    if not isinstance(view, str) or view not in CAMERAS:
        raise ValueError(f"view must be one of {CAMERAS}, got {view!r}")
    return view


def _mm_per_s(speed_m_per_min: float) -> float:
    return speed_m_per_min * 1000.0 / 60.0


def web_order_pairs(rows) -> List[Dict]:
    """The roll as [{md_mm, left, right}] along the web. Every position needs one frame per camera, and positions step
    one frame pitch at a time, so slice k of the roll is k pitches from its start, as the log says."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a non-empty list of manifest rows")
    by_md: Dict[float, Dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not _is_number(row.get("md_offset_mm")) or not isinstance(row.get("file_name"), str):
            raise ValueError(f"a manifest row needs a numeric md_offset_mm and a file_name, got {row!r}")
        camera = _check_view(row.get("camera"))
        files = by_md.setdefault(float(row["md_offset_mm"]), {})
        if camera in files:
            raise ValueError(f"two {camera} frames at {row['md_offset_mm']} mm")
        files[camera] = row["file_name"]
    pitch = frame_geometry()["md_mm"]
    pairs = []
    for md in sorted(by_md):
        if set(by_md[md]) != set(CAMERAS):
            raise ValueError(f"the position at {md} mm does not have a frame for each camera")
        if pairs and abs(md - pairs[-1]["md_mm"] - pitch) > _PITCH_TOLERANCE_MM:
            raise ValueError(f"the roll jumps from {pairs[-1]['md_mm']} mm to {md} mm; positions must be {pitch} mm apart")
        pairs.append({"md_mm": md, **{camera: by_md[md][camera] for camera in CAMERAS}})
    return pairs


def parse_detectors(specs) -> Dict[str, str]:
    """`--detector NAME=DIR` values (a YOLO-Pro size and its ONNX folder), in the order given."""
    detectors: Dict[str, str] = {}
    for spec in specs:
        name, sep, folder = spec.partition("=")
        if not sep or not name or not folder or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError(f"a detector is NAME=DIR with NAME of letters, digits, _ or -, got {spec!r}")
        if name in detectors:
            raise ValueError(f"detector {name!r} is named twice")
        detectors[name] = folder
    return detectors


def check_command(command, placements: Tuple[str, ...] = (), gates: Tuple[str, ...] = (), detectors: Tuple[str, ...] = (),
                  devices: Tuple[str, ...] = ()) -> Dict:
    """One command from the page: exactly one of speed (m/min), pause (bool), step (true), view (camera), or, where
    `placements` are on offer, the splash screen's start {placement, speed, view}, plus the gate where `gates` are on
    offer, the detector where `detectors` are and the device option where `devices` are."""
    if not isinstance(command, Mapping) or len(command) != 1:
        raise ValueError(f"a command is one of speed, pause, step, view or start, got {command!r}")
    (key, value), = command.items()
    if key == "speed":
        return {"speed": _check_speed(value)}
    if key in ("pause", "shade", "heat") and isinstance(value, bool):  # shade, heat: the screen's overlays, not the line
        return {key: value}
    if key == "step" and value is True:
        return {"step": True}
    if key == "view":
        return {"view": _check_view(value)}
    if key == "start" and placements:
        fields = (_START_FIELDS + (("gate",) if gates else ()) + (("detector",) if detectors else ())
                  + (("device",) if devices else ()))
        if not isinstance(value, Mapping) or set(value) != set(fields):
            raise ValueError(f"a start names exactly {fields}, got {value!r}")
        if value["placement"] not in placements:
            raise ValueError(f"placement must be one of {tuple(placements)}, got {value['placement']!r}")
        start = {"placement": value["placement"], "speed": _check_speed(value["speed"]), "view": _check_view(value["view"])}
        if gates:
            if not isinstance(value["gate"], str) or value["gate"] not in gates:
                raise ValueError(f"gate must be one of {tuple(gates)}, got {value['gate']!r}")
            start["gate"] = value["gate"]
        if detectors:
            if not isinstance(value["detector"], str) or value["detector"] not in detectors:
                raise ValueError(f"detector must be one of {tuple(detectors)}, got {value['detector']!r}")
            start["detector"] = value["detector"]
        if devices:
            if not isinstance(value["device"], str) or value["device"] not in devices:
                raise ValueError(f"device must be one of {tuple(devices)}, got {value['device']!r}")
            start["device"] = value["device"]
        return {"start": start}
    raise ValueError(f"bad command {command!r}")


class LineControl:
    """What the operator has set, with a timeline of speed and pause so the belt's travel over any past interval can
    be worked out exactly. Thread-safe: the page's commands arrive on one thread, the driver reads on another. Times
    are on `clock` (monotonic seconds)."""

    def __init__(self, speed_m_per_min: float, view: str = "left", clock: Callable[[], float] = time.monotonic):
        if not callable(clock):
            raise ValueError("clock must be callable")
        speed = _check_speed(speed_m_per_min)
        self._view, self.clock = _check_view(view), clock
        self._step = False
        self._timeline: List[Tuple[float, float, bool]] = [(-math.inf, speed, False)]     # (from, speed, paused)
        self._lock = threading.Lock()

    @property
    def speed(self) -> float:
        with self._lock:
            return self._timeline[-1][1]

    def _paused(self) -> bool:
        return self._timeline[-1][2]

    def running(self) -> bool:
        with self._lock:
            return not self._paused()

    def state(self) -> Dict:
        with self._lock:
            return {"speed_m_per_min": self._timeline[-1][1], "paused": self._paused(), "view": self._view}

    def apply(self, command) -> Dict:
        """Apply one command (see `check_command`) and return the new state. A step is refused while the line runs."""
        (key, value), = check_command(command).items()
        if key in ("shade", "heat"):
            raise ValueError(f"the {key} switch belongs to the screen, not the line")
        with self._lock:
            now = self.clock()
            if key == "speed":
                self._timeline.append((now, value, self._paused()))
            elif key == "pause":
                self._timeline.append((now, self._timeline[-1][1], value))
                self._step = False
            elif key == "step":
                if not self._paused():
                    raise ValueError("a step is only taken while the line is paused")
                self._step = True
            else:
                self._view = value
        return self.state()

    def take_step(self) -> bool:
        """True once per step the operator asked for."""
        with self._lock:
            step, self._step = self._step, False
            return step

    def moving(self, t0: float, t1: float) -> List[Tuple[float, float, float]]:
        """The pieces of [t0, t1] in which the belt moved, as (start, end, mm/s). What happened before `t0` is
        forgotten: the caller asks about later and later intervals."""
        with self._lock:
            first = max(i for i, entry in enumerate(self._timeline) if entry[0] <= t0)
            del self._timeline[:first]
            cuts = {t0, t1}
            cuts.update(t for t, _, _ in self._timeline if t0 < t < t1)
            edges = sorted(cuts)
            pieces = []
            for a, b in zip(edges, edges[1:]):
                _, speed, paused = next(entry for entry in reversed(self._timeline) if entry[0] <= (a + b) / 2)
                if not paused:
                    pieces.append((a, b, _mm_per_s(speed)))
            return pieces


@dataclass(frozen=True)
class TimedGeometry(LineGeometry):
    """A line geometry whose frame times are the moments the belt reached each frame (`capture_s`, seconds after
    `start_time`), for a belt whose speed changes; before a frame is stamped it falls back to the fixed speed. With
    `roll_frames`, the roll loops: slice k is logged at its place on the roll, (k mod roll_frames) pitches from its start."""
    capture_s: Dict[int, float] = field(default_factory=dict, compare=False, repr=False)
    roll_frames: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.roll_frames is not None and (isinstance(self.roll_frames, bool) or not isinstance(self.roll_frames, int) or self.roll_frames < 1):
            raise ValueError(f"roll_frames must be a positive integer, got {self.roll_frames!r}")

    def frame_md_mm(self, frame_index: int) -> float:
        md = super().frame_md_mm(frame_index)
        return md if self.roll_frames is None else (frame_index % self.roll_frames) * self.frame_pitch_mm

    def frame_time(self, frame_index: int) -> datetime:
        seconds = self.capture_s.get(frame_index)
        if seconds is None:                                  # the belt ran k pitches, wherever that is on the roll
            return LineGeometry(self.lane_width_mm, self.belt_speed_m_per_min, self.start_time).frame_time(frame_index)
        return self.start_time + timedelta(seconds=seconds)


class ControlledSource:
    """A driver source whose belt follows `control`. Slice k is due when the belt has carried k frame pitches;
    `load(k)` gives its frames. The driver hands over its clock with `bind_clock` (seconds from the start of the run);
    capture times are on that clock."""

    def __init__(self, load: Callable[[int], Mapping], geometry: TimedGeometry, control: LineControl, count: int):
        if not callable(load):
            raise ValueError("load must be callable")
        if not isinstance(geometry, TimedGeometry):
            raise ValueError(f"geometry must be a TimedGeometry, got {type(geometry).__name__}")
        if not isinstance(control, LineControl):
            raise ValueError(f"control must be a LineControl, got {type(control).__name__}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"count must be a positive integer, got {count!r}")
        self._load, self._geometry, self._control, self._count = load, geometry, control, count
        self._pitch = geometry.frame_pitch_mm
        self._offset: Optional[float] = None             # control clock minus driver clock
        self._next, self._position_mm, self._last, self._due = 0, 0.0, None, None

    def bind_clock(self, now: Callable[[], float]) -> None:
        self._offset = self._control.clock() - now()
        self._last = self._control.clock()

    def _advance(self) -> float:
        """Move the belt up to now (control clock) and fix the next slice's due time once the belt has reached it."""
        if self._offset is None:
            raise ValueError("the source has no clock yet (bind_clock)")
        t = self._control.clock()
        target = self._next * self._pitch
        if self._due is None:
            for a, b, speed in self._control.moving(self._last, t):
                end = self._position_mm + speed * (b - a)
                if end >= target - _PITCH_TOLERANCE_MM:                  # a sleep can land a rounding error short
                    self._due = min(b, a + max(0.0, target - self._position_mm) / speed)
                    end = max(end, target)
                self._position_mm = end
                if self._due is not None:
                    break
            if self._due is None and self._position_mm >= target - _PITCH_TOLERANCE_MM:
                self._due = self._last
            if self._due is None and self._control.take_step():
                self._position_mm, self._due = target, t
        self._last = t
        return t

    def peek_time(self) -> Optional[float]:
        if self._next >= self._count:
            return None
        t = self._advance()
        if self._due is not None:
            return self._due - self._offset
        if not self._control.running():
            return t + POLL_S - self._offset
        return t + (self._next * self._pitch - self._position_mm) / _mm_per_s(self._control.speed) - self._offset

    def remaining(self) -> int:
        return self._count - self._next

    @property
    def taken(self) -> int:
        """How many slices the belt has handed out so far."""
        return self._next

    def now(self) -> float:
        """The time on the driver's clock (the one capture times are on)."""
        if self._offset is None:
            raise ValueError("the source has no clock yet (bind_clock)")
        return self._control.clock() - self._offset

    def take(self) -> Arrival:
        due = self.peek_time()
        if due is None:
            raise ValueError("the source has no more slices")
        if self._due is None:
            raise ValueError(f"slice {self._next} is not due yet")
        k, self._next, self._due = self._next, self._next + 1, None
        self._geometry.capture_s[k] = due
        return Arrival(k, due, self._load(k))


class ReadAhead:
    """`reader(k)` returns `load(k % n)`, keeping up to `ahead` items loaded past the last one asked for, on `workers`
    threads that load side by side (PNG decoding releases the GIL). Items must be asked for in order (0, 1, 2, ...)
    and are handed out in that order whichever finishes first; an error in `load` is raised when that item is asked for."""

    def __init__(self, load: Callable[[int], object], n: int, ahead: int = 8, workers: int = 1):
        if not callable(load):
            raise ValueError("load must be callable")
        for name, value in (("n", n), ("ahead", ahead), ("workers", workers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._load, self._n, self._ahead = load, n, ahead
        self._items: Dict[int, Tuple[object, Optional[BaseException]]] = {}   # loaded, not yet asked for
        self._asked = self._next = 0
        self._closed = False
        self._cond = threading.Condition()
        self._threads = [threading.Thread(target=self._fill, name=f"read-ahead-{i}", daemon=True) for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def _fill(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closed or self._next < self._asked + self._ahead)
                if self._closed:
                    return
                k, self._next = self._next, self._next + 1
            try:
                item = (self._load(k % self._n), None)
            except Exception as exc:
                item = (None, exc)
            with self._cond:
                self._items[k] = item
                self._cond.notify_all()

    def held(self) -> int:
        """Items loaded and not yet asked for."""
        with self._cond:
            return len(self._items)

    def __call__(self, k: int):
        with self._cond:
            if k != self._asked:
                raise ValueError(f"items are read in order: expected {self._asked}, got {k!r}")
            self._cond.wait_for(lambda: k in self._items or self._closed)
            if k not in self._items:
                raise ValueError("the reader is closed")
            value, error = self._items.pop(k)
            self._asked += 1
            self._cond.notify_all()
        if error is not None:
            raise error
        return value

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(5.0)


class StageTimer:
    """Adds up how long each wrapped model call takes, per stage, until `take` hands the totals over and starts again.
    The board's TensorRT calls wait for the GPU before they return, so the time of a call is the time of its work."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock, self._lock = clock, threading.Lock()
        self._totals: Dict[str, List[float]] = {}                       # stage -> [seconds, calls]

    def wrap(self, name: str, fn: Callable) -> Callable:
        if not isinstance(name, str) or not name or not callable(fn):
            raise ValueError(f"a stage needs a name and a callable, got {name!r} and {fn!r}")
        with self._lock:
            self._totals.setdefault(name, [0.0, 0])

        def timed(*args, **kwargs):
            began = self._clock()
            try:
                return fn(*args, **kwargs)
            finally:
                spent = self._clock() - began
                with self._lock:
                    self._totals[name][0] += spent
                    self._totals[name][1] += 1
        return timed

    def take(self) -> Dict:
        with self._lock:
            out = {}
            for name, (seconds, calls) in self._totals.items():
                out[f"{name}_ms"], out[f"{name}_calls"] = seconds * 1000.0, calls
                self._totals[name] = [0.0, 0]
            return out


class AsyncSink:
    """The driver's `on_slice` on its own thread: each call queues `fn(*args)` and returns, so the driver's loop inspects
    the next slice while this one is handed to the screen. Calls run in order; at most `depth` wait, and a further call
    waits for room, so no slice is dropped and memory stays bounded. An error in `fn` is raised by the next call (the
    driver reports it as a 'sink' alarm) and by `close`."""

    def __init__(self, fn: Callable, depth: int = 2):
        if not callable(fn):
            raise ValueError("the sink must be callable")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth!r}")
        self._fn, self._queue = fn, queue.Queue(depth)
        self._error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="slice-sink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            args = self._queue.get()
            if args is None:
                return
            try:
                self._fn(*args)
            except Exception as exc:
                if self._error is None:
                    self._error = exc

    def _raise(self) -> None:
        if self._error is not None:
            exc, self._error = self._error, None
            raise exc

    def __call__(self, *args) -> None:
        if self._closed:
            raise ValueError("the sink is closed")
        self._raise()
        self._queue.put(args)

    def close(self) -> None:
        """Wait for every queued call, stop the thread, and raise an error left by the last calls."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._raise()


def slice_telemetry(k: int, geometry: TimedGeometry, taken: int, now: float, timings: Mapping, status: Optional[Mapping],
                    outcome=None, log_records: Optional[int] = None, mqtt: Optional[Tuple[int, int]] = None) -> Dict:
    """What the page shows about slice k: the web run so far (as an encoder would count it), its place on the roll,
    the time the gate (encoder + head) and YOLO-Pro took and on how many tiles, how far the screen is behind the belt
    (slices handed out since, and seconds since its capture), and the scheduler's state and queue."""
    out = {"slices": k + 1, "web_m": (k + 1) * geometry.frame_pitch_mm / 1000.0, "roll_md_mm": geometry.frame_md_mm(k)}
    if geometry.roll_frames is not None:                # the looping test roll's length
        out["roll_m"] = geometry.roll_frames * geometry.frame_pitch_mm / 1000.0
    for stage in ("gate", "detector"):
        if f"{stage}_ms" in timings:
            out[f"{stage}_ms"], out[f"{stage}_tiles"] = round(timings[f"{stage}_ms"], 1), timings[f"{stage}_calls"]
    out["lag_slices"] = max(0, taken - (k + 1))
    out["lag_s"] = round(max(0.0, now - geometry.capture_s.get(k, now)), 2)
    if status:
        out["state"], out["queue"] = status["state"], status["queue_depth"]
    if outcome is not None:                         # the slice's cascade.SliceResult: the gate's flags, YOLO-Pro's boxed tiles
        out["flagged_tiles"] = sum(t.flagged for r in outcome.results.values() for t in r.tiles)
        out["bbox_tiles"] = len({(camera, d.tile) for camera, r in outcome.results.items() for d in r.detections})
        scores = [t.score for r in outcome.results.values() for t in r.tiles]
        if scores:
            out["gate_max"] = round(max(scores), 1)     # the slice's highest tile score, for the legend's needle
        out["set_aside"] = len(getattr(outcome, "set_aside", ()))   # classes the station cannot have (cascade.STATION_CLASSES)
    if log_records is not None:                     # records written to the defect log so far, and MQTT messages (sent, failed)
        out["log_records"] = log_records
    if mqtt is not None:
        out["mqtt_sent"], out["mqtt_failed"] = mqtt
    return out


ACCELERATORS = ("GPU", "CPU", "NPU")


def _check_runtime(runtime, version, accelerator) -> None:
    if not isinstance(runtime, str) or not runtime or not isinstance(version, str) or not version:
        raise ValueError(f"the runtime needs a name and a version, got {runtime!r} {version!r}")
    if accelerator not in ACCELERATORS:
        raise ValueError(f"accelerator must be one of {ACCELERATORS}, got {accelerator!r}")


def runtime_info(host, runtime: str, version: str, accelerator: str, precision: str) -> Dict:
    """What the HMI says the models run on: the accelerator, the engine runtime and its version, the device (from
    `bench.detect_host`, without the dev-kit wording) and the numeric format the engines were built in."""
    _check_runtime(runtime, version, accelerator)
    device = host.board.replace(" Engineering Reference Developer Kit", "").replace(" Developer Kit", "")
    return {"accelerator": accelerator, "runtime": f"{runtime} {version}", "device": device, "precision": precision,
            "rung": host.rung}


def runtime_panel(device, backend) -> Dict:
    """The HMI header's runtime line on any machine: the backend that loaded, as (runtime, version, accelerator,
    precision), or "runtime not known" when none could be read, always with the device (`device.describe_device`)
    and its rung, so a run off the board is never read as the board's figures."""
    rung = device.rung or "no rung"
    if backend is None:
        return {"accelerator": "—", "runtime": "runtime not known", "device": device.name, "precision": "—", "rung": rung}
    runtime, version, accelerator, precision = backend
    _check_runtime(runtime, version, accelerator)
    return {"accelerator": accelerator, "runtime": f"{runtime} {version}", "device": device.name, "precision": precision,
            "rung": rung}
