"""Checking a machine against the board: which published test frames to run, how a run's gate scores
and detections compare with the board's TensorRT results, and how a speed ladder on the real HMI is read.
`scripts/check_device.py` uses it."""
import re
from typing import Dict, List, Optional, Sequence

SLICE_M = 0.54                                   # metres of web per slice (HMI telemetry)
THRESHOLDS = (20.3522, 25.3)                     # the two gates' thresholds (launcher.hmi_command)
REFERENCE_FRAMES = tuple(f"{folder}/test-test0-{i:05d}.png" for folder in ("A_1080", "B_1080") for i in (0, 70, 140, 210))
_STATUS = re.compile(r"t=\s*([0-9.]+)s\s+([A-Z]+)\s+queue=\s*(\d+)\s+written=(\d+)")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 1.0 if list(a) == list(b) else 0.0


def compare(reference: Sequence[Dict], rows: Sequence[Dict], thresholds=THRESHOLDS) -> Dict:
    """Gate scores and detections of `rows` against the board's `reference`, tile by tile, with the pass verdict."""
    key = lambda r: (r["frame"], tuple(r["tile"]))
    ref = {key(r): r for r in reference}
    got = {key(r): r for r in rows}
    if set(ref) != set(got):
        raise ValueError(f"the runs cover different tiles: {len(set(ref) ^ set(got))} differ")
    abs_diff = rel_diff = 0.0
    flips = {f"{t:g}": 0 for t in thresholds}
    flips_near = True
    matched, worst_score = 0, 0.0
    for k, a in ref.items():
        b = got[k]
        d = abs(a["score"] - b["score"])
        abs_diff, rel_diff = max(abs_diff, d), max(rel_diff, d / max(abs(a["score"]), 1e-9))
        for t in thresholds:
            if (a["score"] > t) != (b["score"] > t):
                flips[f"{t:g}"] += 1
                flips_near &= abs(a["score"] - t) <= 0.01 * t and abs(b["score"] - t) <= 0.01 * t
        free = list(b["dets"])
        for det in a["dets"]:
            best = max(((_iou(det["box"], e["box"]), e) for e in free if e["cls"] == det["cls"]),
                       default=(0.0, None), key=lambda p: p[0])
            if best[1] is not None and best[0] >= 0.9:
                matched += 1
                worst_score = max(worst_score, abs(det["score"] - best[1]["score"]))
                free.remove(best[1])
    n_ref, n_got = sum(len(r["dets"]) for r in reference), sum(len(r["dets"]) for r in rows)
    detections = {"reference": n_ref, "candidate": n_got, "matched": matched, "worst_score_diff": worst_score}
    passed = rel_diff <= 0.01 and flips_near and matched == n_ref == n_got and worst_score <= 0.02
    return {"tiles": len(ref), "max_abs_diff": abs_diff, "max_rel_diff": rel_diff, "flips": flips,
            "flips_near_threshold_only": flips_near, "detections": detections, "pass": passed}


def parse_status(line: str) -> Optional[tuple]:
    """(t seconds, state, queue, slices written) from one HMI status line, None for any other line."""
    m = _STATUS.search(line)
    return (float(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))) if m else None


def slices_needed(speed_m_per_min: float) -> float:
    return speed_m_per_min / 60.0 / SLICE_M


def ladder_result(samples: List[Dict]) -> Optional[float]:
    """The highest speed whose queue stayed empty with slices written at least 95 % as fast as it needs."""
    kept = [s["speed"] for s in samples if s["max_queue"] == 0 and s["rate"] >= 0.95 * s["need"]]
    return max(kept) if kept else None
