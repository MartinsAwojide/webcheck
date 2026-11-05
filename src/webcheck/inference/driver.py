"""The demo driver: runs the scheduler's actions against live inputs.

A source hands out slices at their capture times. The driver admits each one to the scheduler (scheduler.py) and runs
the actions it returns in order, one slice at a time: Inspect calls the injected `inspect(frames, frame_index)`, which
returns the slice's records, and Write sends them to the log. A slice's records reach the log only on its Write action.
Time comes from an injected clock, and the capture times are on the same clock (seconds from the start of the run), so
the latency of a slice is the time from its capture to its records being logged.

A failed inspection or an overload fault in 'fault' mode stops the run and says so in the report; the driver never
drops, skips or resets a slice. In 'block' mode the source is not polled until the scheduler says there is room, and the
slices that fell due in the meantime keep their scheduled capture times, so the wait shows up as latency."""
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from webcheck.inference.defect_log import LineGeometry
from webcheck.inference.scheduler import (
    Admit,
    Alarm,
    BlockSource,
    Failed,
    Inspect,
    Inspected,
    SchedulerConfig,
    UnblockSource,
    Write,
    Written,
    metrics,
    new_state,
    step,
    system_state,
)

STATES = ("NORMAL", "BACKLOGGED", "OVERLOADED", "FAULTED")
_UNFINISHED = ("QUEUED", "INSPECTING", "READY")
DUE_TOLERANCE_S = 1e-6        # a slice within a microsecond of its capture time is due (a sleep can land a rounding error short)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class Arrival:
    """One slice: the frame of each camera, captured `capture_time` seconds after the start of the run."""
    frame_index: int
    capture_time: float
    frames: Mapping


class ReplaySource:
    """Replays stored frames as a line would deliver them: slice k is due when the belt has carried the web k frame
    pitches, so one belt speed (the geometry's) drives both the arrival schedule and the logged timestamps. `slices`
    (a list of {camera: frame}) is cycled until `count` slices have been handed out."""

    def __init__(self, slices: Sequence[Mapping], geometry: LineGeometry, count: int):
        if not isinstance(geometry, LineGeometry):
            raise ValueError(f"geometry must be a LineGeometry, got {type(geometry).__name__}")
        if not isinstance(slices, (list, tuple)) or not slices or not all(isinstance(s, Mapping) and s for s in slices):
            raise ValueError("slices must be a non-empty list of non-empty {camera: frame} mappings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"count must be a positive integer, got {count!r}")
        self._slices, self._geometry, self._count, self._next = list(slices), geometry, count, 0

    def peek_time(self) -> Optional[float]:
        """Capture time of the next slice, or None once all have been handed out."""
        if self._next >= self._count:
            return None
        return self._geometry.frame_md_mm(self._next) / 1000.0 / (self._geometry.belt_speed_m_per_min / 60.0)

    def remaining(self) -> int:
        return self._count - self._next

    def take(self) -> Arrival:
        capture_time = self.peek_time()
        if capture_time is None:
            raise ValueError("the source has no more slices")
        k, self._next = self._next, self._next + 1
        return Arrival(k, capture_time, self._slices[k % len(self._slices)])


def _stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ms = np.asarray(values) * 1e3
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "max": float(ms.max())}


@dataclass(frozen=True)
class DriverReport:
    stopped: str                         # 'done', 'failed' or 'overloaded'
    admitted: int
    written: int
    failed: int
    not_admitted: int                    # still in the source when the run stopped
    unfinished: Tuple[int, ...]          # admitted but not logged when the run stopped
    latency_s: List[float]               # capture to log, per written slice, in frame order
    service_s: List[float]               # time inside `inspect`, per inspected slice
    worst_latency_s: float
    blocks: int                          # times the source was told to wait
    alarms: List[Tuple[str, str]]
    state_seconds: Dict[str, float]
    max_queue_depth: int
    max_oldest_age_s: float
    run_s: float

    def summary(self, ceiling_s: float) -> Dict:
        """The figures worth printing; `share_within_ceiling` is the share of written slices logged within `ceiling_s`."""
        if not _is_number(ceiling_s) or ceiling_s <= 0:
            raise ValueError(f"ceiling_s must be a positive number, got {ceiling_s!r}")
        within = float(np.mean(np.asarray(self.latency_s) <= ceiling_s)) if self.latency_s else None
        return {"stopped": self.stopped, "slices": len(self.latency_s), "latency_ms": _stats(self.latency_s),
                "service_ms": _stats(self.service_s), "share_within_ceiling": within,
                "achieved_slices_per_s": self.written / self.run_s if self.run_s > 0 else None,
                "max_queue_depth": self.max_queue_depth, "max_oldest_age_s": self.max_oldest_age_s,
                "state_seconds": dict(self.state_seconds), "blocks": self.blocks, "alarms": list(self.alarms)}


def _check_arrival(arrival) -> int:
    frames = arrival.frames
    if not isinstance(frames, Mapping) or not frames or not all(isinstance(f, np.ndarray) for f in frames.values()):
        raise ValueError(f"slice {arrival.frame_index!r} needs a non-empty {{camera: array}} mapping of frames")
    return sum(f.nbytes for f in frames.values())


def run_driver(source, inspect: Callable, log, config: SchedulerConfig, clock=None,
               on_status: Optional[Callable] = None, status_every_s: float = 1.0,
               on_slice: Optional[Callable] = None, max_sleep_s: Optional[float] = None) -> DriverReport:
    """Run `source` to the end, or until a failure or an overload fault. `inspect(frames, frame_index)` returns an
    outcome whose `.records` are the slice's records; `log.write(record)` takes them; `on_status(t, metrics)` is called
    about every `status_every_s`. `on_slice(frame_index, frames, outcome)`, if given, is called for each slice after its
    records are logged, so a display can show it (a log with `end_slice(frame_index, outcome)` is told first, so it can
count tiles); it must return quickly, and an exception in it is reported as a 'sink'
    alarm without stopping the run or the log. Frames are held until the sink has seen them. `max_sleep_s`, if given,
    caps each wait for the next arrival, for a source whose schedule can change while the driver waits (hmi.py); a source
    with `bind_clock(now)` is handed the run's clock (seconds from the start of the run) before the first look."""
    if not all(callable(getattr(source, name, None)) for name in ("peek_time", "take", "remaining")):
        raise ValueError("source needs peek_time(), take() and remaining()")
    if not callable(inspect) or not callable(getattr(log, "write", None)):
        raise ValueError("inspect must be callable and log must have write(record)")
    if not isinstance(config, SchedulerConfig):
        raise ValueError(f"config must be a SchedulerConfig, got {type(config).__name__}")
    if on_status is not None and not callable(on_status):
        raise ValueError("on_status must be callable")
    if on_slice is not None and not callable(on_slice):
        raise ValueError("on_slice must be callable")
    if not _is_number(status_every_s) or status_every_s <= 0:
        raise ValueError(f"status_every_s must be a positive number, got {status_every_s!r}")
    if max_sleep_s is not None and (not _is_number(max_sleep_s) or max_sleep_s <= 0):
        raise ValueError(f"max_sleep_s must be a positive number, got {max_sleep_s!r}")
    end_slice = getattr(log, "end_slice", None)
    end_slice = end_slice if callable(end_slice) else None
    clock = clock or MonotonicClock()
    start = clock.now()
    now = lambda: clock.now() - start
    if callable(getattr(source, "bind_clock", None)):
        source.bind_clock(now)

    state, last, pending = new_state(), 0.0, deque()
    frames, records, outcomes = {}, {}, {}
    latency, service, alarms = [], [], []
    state_seconds = dict.fromkeys(STATES, 0.0)
    admitted = blocks = 0
    max_depth, max_age, next_status = 0, 0.0, 0.0

    def apply(event) -> float:
        nonlocal state, last, max_depth, max_age
        t = now()
        state_seconds[system_state(state, t, config)] += t - last
        last = t
        state, actions = step(state, event, t, config)
        m = metrics(state, t, config)
        max_depth, max_age = max(max_depth, m["queue_depth"]), max(max_age, m["oldest_age_s"])
        pending.extend(actions)
        return t

    def tick_status():
        nonlocal next_status
        t = now()
        if on_status is not None and t >= next_status:
            on_status(t, metrics(state, t, config))
            next_status += status_every_s
            if next_status <= t:                 # ticks missed during a long action are skipped, not fired in a burst
                next_status = t + status_every_s

    def admit_due():
        nonlocal admitted
        while not state.blocked and not state.faulted:
            due = source.peek_time()
            if due is None or due > now() + DUE_TOLERANCE_S:
                break
            arrival = source.take()
            frames[arrival.frame_index] = arrival.frames
            admitted += 1
            apply(Admit(arrival.frame_index, arrival.capture_time, _check_arrival(arrival)))

    while True:
        admit_due()
        tick_status()
        while pending:
            action = pending.popleft()
            if isinstance(action, Inspect):
                k, began = action.frame_index, clock.now()
                try:
                    outcome = inspect(frames[k], k)
                    found = list(outcome.records)
                except Exception as exc:
                    del frames[k]
                    apply(Failed(k, str(exc) or repr(exc)))
                    continue
                service.append(clock.now() - began)
                records[k] = found
                if on_slice is None:
                    del frames[k]
                if on_slice is not None or end_slice is not None:
                    outcomes[k] = outcome
                apply(Inspected(k))
            elif isinstance(action, Write):
                for record in records.pop(action.frame_index):
                    log.write(record)
                outcome = outcomes.pop(action.frame_index, None)
                if end_slice is not None:
                    end_slice(action.frame_index, outcome)
                latency.append(apply(Written(action.frame_index)) - action.capture_time)
                if on_slice is not None:
                    k = action.frame_index
                    try:
                        on_slice(k, frames.pop(k), outcome)
                    except Exception as exc:
                        alarms.append(("sink", f"slice {k}: {exc}"))
            elif isinstance(action, Alarm):
                alarms.append((action.kind, action.detail))
            elif isinstance(action, BlockSource):
                blocks += 1
            elif not isinstance(action, UnblockSource):
                raise ValueError(f"unknown action {action!r}")
            admit_due()                          # new arrivals join the back of the queue; they never jump an action
            tick_status()
        due = source.peek_time()
        if state.faulted or due is None:
            break
        wait = max(0.0, due - now())
        clock.sleep(wait if max_sleep_s is None else min(wait, max_sleep_s))

    end = now()
    state_seconds[system_state(state, end, config)] += end - last
    stopped = "done" if not state.faulted else ("failed" if state.failed else "overloaded")
    return DriverReport(
        stopped=stopped, admitted=admitted, written=state.written, failed=state.failed, not_admitted=source.remaining(),
        unfinished=tuple(s.frame_index for s in state.slices if s.status in _UNFINISHED), latency_s=latency,
        service_s=service, worst_latency_s=state.worst_latency_s, blocks=blocks, alarms=alarms,
        state_seconds=state_seconds, max_queue_depth=max_depth, max_oldest_age_s=max_age, run_s=end)
