"""One frame through the cascade: cut six tiles, gate every tile, run the detector only on flagged tiles.
The models are callables, so tests pass fakes and the board passes the TensorRT ones."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from webcheck.inference.detector import NMS_IOU, SCORE_THRESHOLD, _iou, decode_detections
from webcheck.inference.frame import TILE_SIZE, tile_origins, tile_views
from webcheck.inference.gate import MAP_SIZE, GateConfig, gate_score, is_anomalous
from webcheck.inference.preprocess import detector_input, gate_input


DEFAULT_WORKERS = 4


@lru_cache(maxsize=None)
def _pool(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(workers, thread_name_prefix="gate-resize")


@dataclass(frozen=True)
class TileResult:
    row: int
    col: int
    score: float
    flagged: bool
    anomaly_map: Optional[np.ndarray] = field(default=None, compare=False, repr=False)   # 40x40, only when asked for


@dataclass(frozen=True)
class FrameDetection:
    """A detection in frame pixels; `tile` is the (row, col) of the tile that produced it."""
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    tile: Tuple[int, int]


@dataclass(frozen=True)
class FrameResult:
    tiles: Tuple[TileResult, ...]
    detections: Tuple[FrameDetection, ...]

    @property
    def flagged(self) -> bool:
        return any(t.flagged for t in self.tiles)


def _dedup(detections: List[FrameDetection], dedup_iou: float) -> List[FrameDetection]:
    """The tile rows overlap, so one defect can come from two tiles: of two same-label detections from different
    tiles whose frame boxes overlap by more than `dedup_iou`, keep the higher score (the earlier tile on a tie)."""
    keep: List[int] = []
    for i in sorted(range(len(detections)), key=lambda i: -detections[i].score):
        d = detections[i]
        if not any(detections[k].label == d.label and detections[k].tile != d.tile
                   and _iou(_corners(detections[k]), _corners(d)) > dedup_iou for k in keep):
            keep.append(i)
    return [detections[i] for i in sorted(keep)]


def _corners(d: FrameDetection):
    return (d.x1, d.y1, d.x2, d.y2)


def process_frame(frame, gate: Callable, detector: Callable, config: GateConfig,
                  score_threshold: float = SCORE_THRESHOLD, workers: int = DEFAULT_WORKERS,
                  dedup_iou: float = NMS_IOU, gate_inputs: Optional[Callable[..., Sequence]] = None,
                  columns: Optional[Sequence[int]] = None, anomaly_maps: bool = False) -> FrameResult:
    """The gate's 320 resizes run on `workers` threads while the gate works through the tiles in order; the
    models themselves are only ever called from the calling thread. `gate_inputs`, when given, replaces that
    Pillow resize: it takes the frame and returns one gate input per inspected tile, row-major (the GPU resize on the
    board). `columns` limits the inspection to those tile columns (default all three); the other tiles are not
    resized, gated or reported. With `anomaly_maps` each tile keeps a copy of the gate's 40x40 map (the heatmap);
    without it nothing more is done than scoring."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    if isinstance(dedup_iou, bool) or not isinstance(dedup_iou, (int, float)) or not 0 < dedup_iou <= 1:
        raise ValueError(f"dedup_iou must be a number above 0 and at most 1, got {dedup_iou!r}")
    if not isinstance(anomaly_maps, bool):
        raise ValueError(f"anomaly_maps must be True or False, got {anomaly_maps!r}")
    if gate_inputs is not None and not callable(gate_inputs):
        raise ValueError(f"gate_inputs must be callable, got {type(gate_inputs).__name__}")
    tiles = tile_views(frame, columns)
    origins = {(r, c): (x, y) for r, c, x, y in tile_origins(columns)}
    if gate_inputs is None:
        prepared = _pool(workers).map(lambda t: gate_input(np.ascontiguousarray(t[2])), tiles)
    else:
        prepared = gate_inputs(frame)
        if len(prepared) != len(tiles):
            raise ValueError(f"gate_inputs returned {len(prepared)} inputs for {len(tiles)} tiles")
    results, detections = [], []
    for (row, col, tile), g_in in zip(tiles, prepared):
        anomaly_map = gate(g_in)
        score = gate_score(anomaly_map)
        flagged = is_anomalous(score, config)
        kept = np.array(anomaly_map, np.float32).reshape(MAP_SIZE, MAP_SIZE) if anomaly_maps else None   # a copy
        results.append(TileResult(row, col, score, flagged, kept))
        if not flagged:
            continue
        x0, y0 = origins[(row, col)]
        for d in decode_detections(detector(detector_input(tile)), score_threshold):
            detections.append(FrameDetection(d.label, d.score, x0 + d.x1 * TILE_SIZE, y0 + d.y1 * TILE_SIZE,
                                             x0 + d.x2 * TILE_SIZE, y0 + d.y2 * TILE_SIZE, (row, col)))
    return FrameResult(tuple(results), tuple(_dedup(detections, dedup_iou)))
