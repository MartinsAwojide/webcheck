"""A defect in the overlap of the two cameras is logged once.

Detections are compared in web millimetres. Two detections of the same label from different cameras are one defect
when, after each box is grown by DEDUP_GROW_MM on every side, the intersection is at least DEDUP_MIN_IOS of the
smaller box (IoU would lose a thin streak to a 1 mm shift and a defect cut at an image edge to its whole copy). The
untruncated detection is kept, else the higher confidence, else the left camera; it is kept whole and drops every
matching detection of the other camera. A detection only one camera saw is kept. The three numbers are assumed and
fixed; calibrating them on the val pairs is deferred."""
from dataclasses import dataclass
from typing import List, Tuple

from webcheck.inference.frame import FRAME_WIDTH, TILE_COLS, TILE_SIZE, tile_origins

DEDUP_GROW_MM = 2.0
DEDUP_MIN_IOS = 0.5
TRUNCATION_MARGIN_PX = 2.0

CAMERAS = ("left", "right")


@dataclass(frozen=True)
class WebDetection:
    """A detection as a box in web millimetres: md along the web, cd across it."""
    label: str
    score: float
    camera: str
    frame_index: int
    md0: float
    md1: float
    cd0: float
    cd1: float
    truncated: bool          # the box touches an edge of the tile that produced it, so the defect may be larger
    tile: Tuple[int, int]


def web_detections(result, frame_index: int, camera: str, geometry) -> List[WebDetection]:
    """The detections of one camera's frame (a `pipeline.FrameResult`) as web-millimetre boxes, in detection order. The
    image rows run against the web direction: row 0 is the far end of the frame's span along the web."""
    md_end = geometry.frame_md_mm(frame_index) + geometry.frame_pitch_mm       # image row 0 is the far end of the frame
    cd_offset, scale = geometry.cd_offset_mm(camera), geometry.mm_per_px
    origin = {(row, col): (x, y) for row, col, x, y in tile_origins()}
    margin, far = TRUNCATION_MARGIN_PX, TILE_SIZE - TRUNCATION_MARGIN_PX
    out = []
    for d in result.detections:
        x0, y0 = origin[d.tile]
        truncated = d.x1 - x0 <= margin or d.y1 - y0 <= margin or d.x2 - x0 >= far or d.y2 - y0 >= far
        out.append(WebDetection(d.label, d.score, camera, frame_index, md_end - d.y2 * scale, md_end - d.y1 * scale,
                                cd_offset + d.x1 * scale, cd_offset + d.x2 * scale, truncated, d.tile))
    return out


def _overlap_over_smaller(a: WebDetection, b: WebDetection) -> float:
    g = DEDUP_GROW_MM
    width = min(a.cd1, b.cd1) - max(a.cd0, b.cd0) + 2 * g
    height = min(a.md1, b.md1) - max(a.md0, b.md0) + 2 * g
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((d.cd1 - d.cd0 + 2 * g) * (d.md1 - d.md0 + 2 * g) for d in (a, b))
    return width * height / smaller


def _same_defect(a: WebDetection, b: WebDetection) -> bool:
    return a.camera != b.camera and a.label == b.label and _overlap_over_smaller(a, b) >= DEDUP_MIN_IOS


def deduplicate(detections) -> List[WebDetection]:
    """The detections of one frame index from both cameras, with each defect seen twice reduced to one detection:
    the left camera's survivors first, then the right-only ones, each in input order."""
    dets = list(detections)
    if len({d.frame_index for d in dets}) > 1:
        raise ValueError("only detections of one frame index can be de-duplicated together")
    if any(d.camera not in CAMERAS for d in dets):
        raise ValueError(f"camera must be one of {CAMERAS}")
    best_first = sorted(range(len(dets)), key=lambda i: (dets[i].truncated, -dets[i].score, CAMERAS.index(dets[i].camera), i))
    kept = []
    for i in best_first:
        if not any(_same_defect(dets[k], dets[i]) for k in kept):
            kept.append(i)
    return [dets[i] for i in sorted(kept, key=lambda i: (CAMERAS.index(dets[i].camera), i))]


@dataclass(frozen=True)
class CameraBox:
    """A kept detection as a box in one camera's full-frame pixels; `source` is the camera that found it."""
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str


def camera_boxes(detections, frame_index: int, camera: str, geometry) -> List[CameraBox]:
    """The kept detections of one frame index (`deduplicate`'s output) drawn in `camera`'s frame, in order: each web box
    is shifted into that camera's pixels and clipped to the frame; one the camera cannot see is left out."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}, got {camera!r}")
    dets = list(detections)
    if any(d.frame_index != frame_index for d in dets):
        raise ValueError(f"only detections of frame {frame_index} can be drawn in it")
    md_end = geometry.frame_md_mm(frame_index) + geometry.frame_pitch_mm       # image row 0 is the far end of the frame
    offset, scale = geometry.cd_offset_mm(camera), geometry.mm_per_px
    out = []
    for d in dets:
        x1, x2 = (d.cd0 - offset) / scale, (d.cd1 - offset) / scale
        if x2 <= 0 or x1 >= FRAME_WIDTH:
            continue
        out.append(CameraBox(d.label, d.score, max(x1, 0.0), (md_end - d.md1) / scale, min(x2, float(FRAME_WIDTH)),
                             (md_end - d.md0) / scale, d.camera))
    return out


def inspect_columns(camera: str, geometry) -> Tuple[int, ...]:
    """The tile columns `camera` must inspect. The left camera inspects all of them; a later camera only the columns
    that reach past the left camera's far edge, because the web under its other columns is already inspected there
    (a defect cut by a tile edge of the left camera is then seen in two pieces)."""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}, got {camera!r}")
    if camera == CAMERAS[0]:
        return tuple(range(TILE_COLS))
    scale = geometry.mm_per_px
    left_end = geometry.cd_offset_mm(CAMERAS[0]) + FRAME_WIDTH * scale
    offset = geometry.cd_offset_mm(camera)
    return tuple(c for c in range(TILE_COLS) if offset + (c + 1) * TILE_SIZE * scale > left_end + 1e-6)
