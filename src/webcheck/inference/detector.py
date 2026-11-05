"""YOLO-Pro output decoding. The exported ONNX gives 1x8400x10: four box corners (x1, y1, x2, y2 as
fractions of the tile) then six class probabilities (already in 0..1). A candidate takes its best class;
scores at or above the threshold are kept; class-aware greedy NMS removes duplicates. Matches Edge Impulse
Studio's detections on the val tiles."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

CLASSES = ("bubbles", "dot_skip", "mottling", "silvering", "streaks", "tunneling")
N_CANDIDATES = 8400
NMS_IOU = 0.45
SCORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xywh(self, size: float) -> Tuple[float, float, float, float]:
        """Top-left x, y, width, height in pixels of a size x size tile."""
        return (self.x1 * size, self.y1 * size, (self.x2 - self.x1) * size, (self.y2 - self.y1) * size)


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_detections(output, score_threshold: float = SCORE_THRESHOLD, nms_iou: float = NMS_IOU) -> List[Detection]:
    out = np.asarray(output)
    if out.shape != (1, N_CANDIDATES, 4 + len(CLASSES)):
        raise ValueError(f"detector output must be 1x{N_CANDIDATES}x{4 + len(CLASSES)}, got {out.shape}")
    if np.isnan(out).any():
        raise ValueError("detector output contains NaN")
    a = out[0]
    probs = a[:, 4:]
    cls = probs.argmax(1)
    best = probs.max(1)
    boxes = np.clip(a[:, :4], 0.0, 1.0)
    order = np.nonzero(best >= score_threshold)[0]
    order = order[np.argsort(-best[order], kind="stable")]
    keep: List[int] = []
    for i in order:
        if all(not (cls[i] == cls[k] and _iou(boxes[i], boxes[k]) > nms_iou) for k in keep):
            keep.append(int(i))
    return [Detection(CLASSES[cls[i]], float(best[i]), *(float(v) for v in boxes[i])) for i in keep]
