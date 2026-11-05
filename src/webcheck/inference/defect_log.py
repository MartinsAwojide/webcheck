"""The defect log. Each YOLO-Pro detection becomes one record `{defect_type, confidence, timestamp, location, size}` in a
JSON Lines file. `size` is the box area in mm² (pixels to mm² is one linear scale); `location` is an encoder-proxy
position because the prototype has no encoder: the frame's place along the web is its index times the frame length, and
the camera adds its cross-web offset. Image rows run against the web direction, so row 0 is the frame's far end."""
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Sequence

from webcheck.datagen.scene import FRAME_MM_PER_PX, bank_cd_offset_mm, frame_geometry
from webcheck.inference.detector import CLASSES

LOG_FIELDS = ("defect_type", "confidence", "timestamp", "location", "size")


def _number(value, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number, got {value!r}")
    return float(value)


def _frame_index(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"frame index must be a non-negative integer, got {value!r}")
    return value


def _aware(value, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a datetime with a time zone, got {value!r}")
    return value


@dataclass(frozen=True)
class LineGeometry:
    """The simulated line: SKU lane width (per-SKU configuration), belt speed and the capture time of frame 0."""
    lane_width_mm: float
    belt_speed_m_per_min: float
    start_time: datetime

    def __post_init__(self):
        _number(self.lane_width_mm, "lane_width_mm", positive=True)
        _number(self.belt_speed_m_per_min, "belt_speed_m_per_min", positive=True)
        _aware(self.start_time, "start_time")

    @property
    def mm_per_px(self) -> float:
        return FRAME_MM_PER_PX

    @property
    def frame_pitch_mm(self) -> float:
        return frame_geometry()["md_mm"]

    def frame_md_mm(self, frame_index: int) -> float:
        """Encoder proxy: how far along the web frame `frame_index` starts."""
        return _frame_index(frame_index) * self.frame_pitch_mm

    def frame_time(self, frame_index: int) -> datetime:
        """Capture time of the frame: the belt covers its start position at the belt speed."""
        seconds = self.frame_md_mm(frame_index) / 1000.0 / (self.belt_speed_m_per_min / 60.0)
        return self.start_time + timedelta(seconds=seconds)

    def cd_offset_mm(self, camera: str) -> float:
        return bank_cd_offset_mm(camera, self.lane_width_mm)


def box_area_mm2(x1: float, y1: float, x2: float, y2: float, mm_per_px: float = FRAME_MM_PER_PX) -> float:
    for value in (x1, y1, x2, y2):
        _number(value, "box coordinate")
    _number(mm_per_px, "mm_per_px", positive=True)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"box corners are reversed: ({x1}, {y1}, {x2}, {y2})")
    return (x2 - x1) * (y2 - y1) * mm_per_px ** 2


@dataclass(frozen=True)
class DefectRecord:
    defect_type: str
    confidence: float
    timestamp: datetime
    md_mm: float
    cd_mm: float
    size_value: float
    size_unit: str

    def __post_init__(self):
        if self.defect_type not in CLASSES:
            raise ValueError(f"defect_type must be one of {CLASSES}, got {self.defect_type!r}")
        if not 0.0 <= _number(self.confidence, "confidence") <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence!r}")
        _aware(self.timestamp, "timestamp")
        _number(self.md_mm, "md_mm")
        _number(self.cd_mm, "cd_mm")
        if _number(self.size_value, "size_value") < 0:
            raise ValueError(f"size_value must not be negative, got {self.size_value!r}")
        if not isinstance(self.size_unit, str) or not self.size_unit:
            raise ValueError(f"size_unit must be a non-empty string, got {self.size_unit!r}")

    def to_dict(self) -> dict:
        utc = self.timestamp.astimezone(timezone.utc)
        stamp = utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"
        return {"defect_type": self.defect_type, "confidence": self.confidence, "timestamp": stamp,
                "location": {"md_mm": self.md_mm, "cd_mm": self.cd_mm},
                "size": {"value": self.size_value, "unit": self.size_unit}}


def records_for_frame(result, frame_index: int, camera: str, geometry: LineGeometry) -> List[DefectRecord]:
    """One record per detection of one camera's frame (a `pipeline.FrameResult`), in detection order."""
    md_end, cd_offset = geometry.frame_md_mm(frame_index) + geometry.frame_pitch_mm, geometry.cd_offset_mm(camera)
    stamp, scale = geometry.frame_time(frame_index), geometry.mm_per_px
    return [DefectRecord(d.label, d.score, stamp, md_end - (d.y1 + d.y2) / 2 * scale, cd_offset + (d.x1 + d.x2) / 2 * scale,
                         box_area_mm2(d.x1, d.y1, d.x2, d.y2), "mm2") for d in result.detections]


def records_from_web(detections: Sequence, geometry: LineGeometry) -> List[DefectRecord]:
    """One record per `overlap.WebDetection`, each stamped with the capture time of its own frame."""
    return [DefectRecord(d.label, d.score, geometry.frame_time(d.frame_index), (d.md0 + d.md1) / 2, (d.cd0 + d.cd1) / 2,
                         (d.md1 - d.md0) * (d.cd1 - d.cd0), "mm2") for d in detections]


class DefectLogWriter:
    """Appends one JSON line per record and flushes it, so a crash loses at most the record being written."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, record: DefectRecord) -> None:
        if self._file is None:
            raise ValueError("the log is closed")
        if not isinstance(record, DefectRecord):
            raise ValueError(f"only DefectRecord can be logged, got {type(record).__name__}")
        self._file.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
