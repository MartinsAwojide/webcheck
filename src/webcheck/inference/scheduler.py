"""The queue and state machine behind the cascade.

One board serves both cameras, so slices can arrive faster than they are inspected. YOLO-Pro is always run, whatever the
latency: nothing is dropped or degraded, the system says it is behind. The unit of work is a slice (one frame index, the
frame of each camera). `step(state, event, now, config) -> (state, actions)` is a pure function with an injected clock;
a thin interpreter runs the actions on the board.

Slice states: QUEUED, INSPECTING, READY, LOGGED, or FAILED. One slice is inspected at a time, in frame-index order. The
system state is derived, never stored: FAULTED, OVERLOADED (queued bytes at the limit), BACKLOGGED (the oldest unfinished
slice is older than the latency ceiling), else NORMAL. At the byte limit nothing is discarded: in 'block' mode (file
replay) the source is told to wait until there is room; in 'fault' mode (a real line) the system faults and refuses new
slices until a reset."""
import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

HISTORY = 256      # finished slices whose state can still be asked for; older ones are forgotten
_OPEN = ("QUEUED", "INSPECTING")
_UNFINISHED = ("QUEUED", "INSPECTING", "READY")
_FINISHED = ("LOGGED", "FAILED")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SchedulerConfig:
    max_latency_s: float           # the latency ceiling a slice may take from capture to log before the system is BACKLOGGED
    max_queue_bytes: int           # queued frame bytes at which the system is OVERLOADED
    on_overload: str = "block"     # 'block': tell the source to wait; 'fault': fault the line

    def __post_init__(self):
        if not _is_number(self.max_latency_s) or self.max_latency_s <= 0:
            raise ValueError(f"max_latency_s must be a positive number, got {self.max_latency_s!r}")
        if not _is_int(self.max_queue_bytes) or self.max_queue_bytes <= 0:
            raise ValueError(f"max_queue_bytes must be a positive integer, got {self.max_queue_bytes!r}")
        if self.on_overload not in ("block", "fault"):
            raise ValueError(f"on_overload must be 'block' or 'fault', got {self.on_overload!r}")


# events
@dataclass(frozen=True)
class Admit:
    frame_index: int
    capture_time: float
    nbytes: int


@dataclass(frozen=True)
class Inspected:
    frame_index: int


@dataclass(frozen=True)
class Failed:
    frame_index: int
    reason: str


@dataclass(frozen=True)
class Written:
    frame_index: int


@dataclass(frozen=True)
class Reset:
    pass


# actions
@dataclass(frozen=True)
class Inspect:
    frame_index: int


@dataclass(frozen=True)
class Write:
    frame_index: int
    capture_time: float


@dataclass(frozen=True)
class BlockSource:
    pass


@dataclass(frozen=True)
class UnblockSource:
    pass


@dataclass(frozen=True)
class Alarm:
    kind: str
    detail: str


ACTIONS = (Inspect, Write, BlockSource, UnblockSource, Alarm)
_OVERLOADED = Alarm("overloaded", "queued bytes at the limit")


@dataclass(frozen=True)
class _Slice:
    frame_index: int
    capture_time: float
    nbytes: int
    status: str


@dataclass(frozen=True)
class SchedulerState:
    slices: Tuple[_Slice, ...] = ()
    faulted: bool = False
    blocked: bool = False          # the source has been told to wait
    written: int = 0
    failed: int = 0
    worst_latency_s: float = 0.0   # largest capture-to-log time so far
    last_now: float = 0.0
    last_admitted: int = -1


def new_state() -> SchedulerState:
    return SchedulerState()


def _queued_bytes(state: SchedulerState) -> int:
    return sum(s.nbytes for s in state.slices if s.status in _OPEN)


def _oldest_age(state: SchedulerState, now: float) -> float:
    return max((now - s.capture_time for s in state.slices if s.status in _UNFINISHED), default=0.0)


def _find(state: SchedulerState, frame_index) -> Optional[_Slice]:
    return next((s for s in state.slices if s.frame_index == frame_index), None)


def slice_state(state: SchedulerState, frame_index: int) -> Optional[str]:
    found = _find(state, frame_index)
    return found.status if found else None


def _require(state: SchedulerState, frame_index, status: str) -> _Slice:
    found = _find(state, frame_index)
    if found is None or found.status != status:
        raise ValueError(f"slice {frame_index!r} is {found.status if found else 'unknown'}, not {status}")
    return found


def _with_status(state: SchedulerState, frame_index: int, status: str) -> SchedulerState:
    slices = tuple(replace(s, status=status) if s.frame_index == frame_index else s for s in state.slices)
    finished = [s for s in slices if s.status in _FINISHED]
    if len(finished) > HISTORY:
        forgotten = {s.frame_index for s in finished[:len(finished) - HISTORY]}
        slices = tuple(s for s in slices if s.frame_index not in forgotten)
    return replace(state, slices=slices)


def _start_next(state: SchedulerState):
    if any(s.status == "INSPECTING" for s in state.slices):
        return state, ()
    waiting = next((s for s in state.slices if s.status == "QUEUED"), None)
    if waiting is None:
        return state, ()
    return _with_status(state, waiting.frame_index, "INSPECTING"), (Inspect(waiting.frame_index),)


def _release_source(state: SchedulerState, actions, config: SchedulerConfig):
    """Tell a waiting source to go on once the queue is strictly below the byte limit."""
    if state.blocked and _queued_bytes(state) < config.max_queue_bytes:
        return replace(state, blocked=False), tuple(actions) + (UnblockSource(),)
    return state, tuple(actions)


def system_state(state: SchedulerState, now: float, config: SchedulerConfig) -> str:
    if state.faulted:
        return "FAULTED"
    if _queued_bytes(state) >= config.max_queue_bytes:
        return "OVERLOADED"
    if _oldest_age(state, now) > config.max_latency_s:
        return "BACKLOGGED"
    return "NORMAL"


def metrics(state: SchedulerState, now: float, config: SchedulerConfig) -> Dict:
    return {"state": system_state(state, now, config),
            "queue_depth": sum(1 for s in state.slices if s.status in _OPEN),
            "queue_bytes": _queued_bytes(state),
            "oldest_age_s": _oldest_age(state, now),
            "written": state.written, "failed": state.failed, "max_latency_s": state.worst_latency_s}


def step(state: SchedulerState, event, now: float, config: SchedulerConfig):
    """Apply one event at time `now`; returns the new state and the actions to run, in order."""
    if not _is_number(now) or now < state.last_now:
        raise ValueError(f"time must not go backwards: {now!r} after {state.last_now!r}")
    if isinstance(event, Reset):
        return _reset(state, now, config)
    state = replace(state, last_now=now)
    if isinstance(event, Admit):
        return _admit(state, event, config)
    if isinstance(event, Inspected):
        return _inspected(state, event, config)
    if isinstance(event, Failed):
        return _failed(state, event)
    if isinstance(event, Written):
        return _written(state, event, now)
    raise ValueError(f"unknown event {event!r}")


def _admit(state: SchedulerState, event: Admit, config: SchedulerConfig):
    if state.faulted:
        raise ValueError("the line is faulted; reset it before admitting more slices")
    if not _is_int(event.frame_index) or event.frame_index <= state.last_admitted:
        raise ValueError(f"frame index must be an integer above {state.last_admitted}, got {event.frame_index!r}")
    if not _is_number(event.capture_time):
        raise ValueError(f"capture_time must be a finite number, got {event.capture_time!r}")
    if not _is_int(event.nbytes) or event.nbytes <= 0:
        raise ValueError(f"nbytes must be a positive integer, got {event.nbytes!r}")
    queued = _Slice(event.frame_index, float(event.capture_time), event.nbytes, "QUEUED")
    state, actions = _start_next(replace(state, last_admitted=event.frame_index, slices=state.slices + (queued,)))
    if _queued_bytes(state) < config.max_queue_bytes:
        return state, actions
    if config.on_overload == "fault":
        return replace(state, faulted=True), actions + (_OVERLOADED,)
    if state.blocked:
        return state, actions
    return replace(state, blocked=True), actions + (BlockSource(), _OVERLOADED)


def _inspected(state: SchedulerState, event: Inspected, config: SchedulerConfig):
    captured = _require(state, event.frame_index, "INSPECTING").capture_time
    state = _with_status(state, event.frame_index, "READY")
    actions = (Write(event.frame_index, captured),)
    if not state.faulted:
        state, started = _start_next(state)
        actions += started
    if state.faulted:
        return state, actions
    return _release_source(state, actions, config)


def _failed(state: SchedulerState, event: Failed):
    _require(state, event.frame_index, "INSPECTING")
    state = _with_status(state, event.frame_index, "FAILED")
    return replace(state, failed=state.failed + 1, faulted=True), (Alarm("failed", event.reason),)


def _written(state: SchedulerState, event: Written, now: float):
    captured = _require(state, event.frame_index, "READY").capture_time
    state = _with_status(state, event.frame_index, "LOGGED")
    return replace(state, written=state.written + 1, worst_latency_s=max(state.worst_latency_s, now - captured)), ()


def _reset(state: SchedulerState, now: float, config: SchedulerConfig):
    if not state.faulted:
        return state, ()
    state, started = _start_next(replace(state, faulted=False, last_now=now))
    return _release_source(state, started, config)
