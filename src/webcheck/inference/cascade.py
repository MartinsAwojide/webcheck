"""The cascade writes the defect log, one MD position (a slice) at a time. A slice is one frame index with
the frame of each camera of the bank. Each camera's frame goes through `process_frame` (the gate scores every tile,
YOLO-Pro runs on every flagged tile whatever the latency); the detections of both cameras are then de-duplicated in
web millimetres and the survivors are written in one go. Nothing is written until the whole slice has been inspected."""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from webcheck.inference.defect_log import DefectRecord, records_from_web
from webcheck.inference.detector import CLASSES
from webcheck.inference.frame import tile_columns, tile_views
from webcheck.inference.overlap import CAMERAS, CameraBox, WebDetection, camera_boxes, deduplicate, web_detections
from webcheck.inference.pipeline import FrameResult, process_frame


@dataclass(frozen=True)
class CameraInspection:
    camera: str
    frame_index: int
    result: FrameResult
    detections: List[WebDetection]


@dataclass(frozen=True)
class RecordSource:
    """Where a logged record came from: the camera whose detection was kept, the gate tile that produced it, that
    tile's gate score, and the box in that camera's full-frame pixels (x1, y1, x2, y2)."""
    camera: str
    tile: Tuple[int, int]
    gate_score: Optional[float]
    box: Tuple[float, float, float, float]


# The classes each station can have: placement A is inspected after printing, before lamination, so only print defects
# (its ground truth: dot_skip, streaks, mottling); placement B, after lamination, can have all six.
STATION_CLASSES = {"A": ("dot_skip", "mottling", "streaks"), "B": tuple(CLASSES)}


@dataclass(frozen=True)
class SliceResult:
    frame_index: int
    results: Dict[str, FrameResult]
    records: List[DefectRecord]
    boxes: Dict[str, List[CameraBox]] = field(default_factory=dict)   # the logged detections, drawn in each camera's pixels
    sources: Tuple[RecordSource, ...] = ()                             # one per record, in record order
    set_aside: Tuple[str, ...] = ()                                    # classes found that the station cannot have

    @property
    def flagged(self) -> bool:
        return any(r.flagged for r in self.results.values())


def inspect_frame(frame, frame_index: int, camera: str, gate, detector, config, geometry, **options) -> CameraInspection:
    """One camera's frame through the cascade; returns its detections in web millimetres and writes nothing."""
    geometry.frame_md_mm(frame_index)       # a bad index or camera is refused before any model runs
    geometry.cd_offset_mm(camera)
    result = process_frame(frame, gate, detector, config, **options)
    return CameraInspection(camera, frame_index, result, web_detections(result, frame_index, camera, geometry))


def inspect_slice(frames: Mapping, frame_index: int, gate, detector, config, geometry, log,
                  camera_options: Optional[Mapping] = None, classes: Optional[Tuple[str, ...]] = None,
                  **options) -> SliceResult:
    """Inspect the frames of `frame_index` (left camera first), drop each defect the two cameras both saw, and write the
    survivors to `log`. A failure in either camera raises before anything is written. `options` go to `process_frame`;
    `camera_options` maps a camera to options that replace them for that camera (the right camera's `columns`). With
    `classes` (STATION_CLASSES), a defect of another class is set aside: not logged, boxed or counted, only reported."""
    if not isinstance(frames, Mapping) or not frames or set(frames) - set(CAMERAS):
        raise ValueError(f"frames must map some of {CAMERAS} to frames, got {frames!r}")
    camera_options = {} if camera_options is None else camera_options
    if not isinstance(camera_options, Mapping) or set(camera_options) - set(CAMERAS) \
            or not all(isinstance(v, Mapping) for v in camera_options.values()):
        raise ValueError(f"camera_options must map cameras of {CAMERAS} to option mappings, got {camera_options!r}")
    for options_of_camera in camera_options.values():
        tile_columns(options_of_camera.get("columns"))              # bad columns are refused before the left camera runs
    geometry.frame_md_mm(frame_index)
    for frame in frames.values():
        tile_views(frame)                                           # a malformed frame is refused before any model runs
    inspections = [inspect_frame(frames[camera], frame_index, camera, gate, detector, config, geometry,
                                 **{**options, **camera_options.get(camera, {})})
                   for camera in CAMERAS if camera in frames]
    found = deduplicate([d for ins in inspections for d in ins.detections])
    survivors = [d for d in found if classes is None or d.label in classes]
    set_aside = tuple(d.label for d in found if classes is not None and d.label not in classes)
    records = records_from_web(survivors, geometry)
    for record in records:
        log.write(record)
    boxes = {ins.camera: camera_boxes(survivors, frame_index, ins.camera, geometry) for ins in inspections}
    results = {ins.camera: ins.result for ins in inspections}
    return SliceResult(frame_index, results, records, boxes, tuple(_source(d, results, geometry) for d in survivors),
                       set_aside)


def _source(d: WebDetection, results: Mapping[str, FrameResult], geometry) -> RecordSource:
    score = next((t.score for t in results[d.camera].tiles if (t.row, t.col) == d.tile), None)
    (box,) = camera_boxes([d], d.frame_index, d.camera, geometry)          # its own camera always sees it
    return RecordSource(d.camera, d.tile, score, (box.x1, box.y1, box.x2, box.y2))
