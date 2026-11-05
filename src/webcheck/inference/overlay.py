"""The picture the demo screen shows for one slice.

`snapshot` takes the light, picklable copy of a finished slice that the display needs: each camera's frame at half size,
the gate's tiles, the detections in full-frame pixels and the logged records. `render_annotated` draws it as one
960x1080 picture, the left camera's frame above the right camera's, with a box and label strip per detection in a colour
per defect type, and the tile columns that gating left uninspected shaded. Pure NumPy and PIL; no models. The boxes are
each camera's own detections, before the two cameras are de-duplicated; the defect panel beside the picture lists the
de-duplicated records. For the HMI a snapshot can also keep one camera at full size (`full`), which
`render_camera` draws as a 1920x1080 picture with the same shading; its boxes are the slice's logged (de-duplicated)
detections in that camera's pixels, so a defect the other camera found is boxed too, tagged (L) or (R).

The HMI picture can also carry the gate's anomaly heatmap (`render_camera(heat_threshold=...)`): each inspected tile's
40x40 map, one cell per 16x16 frame pixels, the higher value where the tile rows overlap, blended under the boxes. The
scale is fixed and relative to the gate threshold T: a cell below HEAT_FLOOR x T is left clear; up to T it is
HEAT_BELOW_COLOR, fading in linearly to HEAT_ALPHA_BELOW; from T (flagged, as the gate flags a tile) it is
HEAT_AT_COLOR at HEAT_ALPHA_ABOVE, running to HEAT_TOP_COLOR at HEAT_TOP x T and staying there."""
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webcheck.inference.frame import FRAME_HEIGHT, FRAME_WIDTH, TILE_COLS, TILE_SIZE, tile_origins
from webcheck.inference.gate import MAP_SIZE
from webcheck.inference.overlap import CAMERAS

SCALE = 2                                            # the picture is the frames at half size
HALF_HEIGHT, HALF_WIDTH = FRAME_HEIGHT // SCALE, FRAME_WIDTH // SCALE
CANVAS_WIDTH, CANVAS_HEIGHT = HALF_WIDTH, HALF_HEIGHT * len(CAMERAS)
BACKGROUND = (24, 24, 24)
SHADE = 0.5                                          # brightness kept in a tile column that was not inspected
LABEL_HEIGHT = 13
BOX_PAD = 8                                          # HMI: px a box is grown on every side, so it clears the defect
BOX_LINE = 4                                         # HMI: box line width in px
LABEL_SCALE = 3                                      # HMI: the label strip and its text at three times the size

CLASS_COLORS = {                                     # colour-blind-safe set
    "bubbles": (79, 179, 255),
    "dot_skip": (255, 200, 0),
    "mottling": (154, 92, 255),
    "silvering": (0, 220, 160),
    "streaks": (255, 70, 70),
    "tunneling": (255, 95, 210),
}
_FALLBACK_COLOR = (255, 255, 255)

HEAT_FLOOR = 0.8                                     # x threshold: cells below are left clear
HEAT_TOP = 1.5                                       # x threshold: the colour stops changing here
HEAT_BELOW_COLOR = (255, 236, 120)                   # pale yellow below the threshold
HEAT_AT_COLOR = (255, 96, 0)                         # orange-red at the threshold
HEAT_TOP_COLOR = (220, 0, 110)                       # crimson from HEAT_TOP x threshold
HEAT_ALPHA_BELOW = 0.35                              # opacity reached just below the threshold
HEAT_ALPHA_ABOVE = 0.6                               # opacity from the threshold up
CROP_SIZE = 160                                      # HMI defect crop: at most this many pixels a side
CROP_MIN = 96                                        # the window never shows less of the frame than this
CROP_MAX = 480                                       # nor more
CROP_MARGIN = 1.6                                    # the window is this many times the box's longer side


@dataclass
class Snapshot:
    frame_index: int
    frames: Dict[str, np.ndarray]                    # camera -> half-size RGB frame
    tiles: Dict[str, List[Tuple]]                    # camera -> [(row, col, score, flagged)]
    detections: Dict[str, List[Tuple]]               # camera -> [(label, score, x1, y1, x2, y2)] in full-frame pixels
    records: List[Dict]                              # the slice's logged records as JSON-ready dicts
    full: Dict[str, np.ndarray] = field(default_factory=dict)        # camera -> full-size RGB frame (HMI only)
    boxes: Dict[str, List] = field(default_factory=dict)             # full-size camera -> the slice's logged boxes in it
    plain: Dict[str, np.ndarray] = field(default_factory=dict)       # camera -> full-size frame with nothing drawn (/web pages)
    heat: Dict[str, List[Tuple]] = field(default_factory=dict)       # full-size camera -> [(row, col, 40x40 map)] when kept


def snapshot(frame_index: int, frames: Mapping, result, full: Optional[str] = None, plain: Tuple[str, ...] = ()) -> Snapshot:
    """Copy what the display needs from a finished slice (`result` is a `cascade.SliceResult`); `full` names a camera
    whose frame is also kept at full size, `plain` the cameras whose full-size frame is kept to be shown as it is. The
    full-size camera's anomaly maps are kept if its tiles carry them (`process_frame(anomaly_maps=True)`)."""
    if not isinstance(frames, Mapping) or not frames or set(frames) - set(CAMERAS):
        raise ValueError(f"frames must map some of {CAMERAS} to frames, got {frames!r}")
    for camera, frame in frames.items():
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise ValueError(f"the {camera} frame must be a {FRAME_HEIGHT}x{FRAME_WIDTH}x3 uint8 array")
    if full is not None and (not isinstance(full, str) or full not in frames):
        raise ValueError(f"full must name a camera of the slice ({sorted(frames)}), got {full!r}")
    if not isinstance(plain, tuple) or len(set(plain)) != len(plain) or not all(isinstance(c, str) and c in frames for c in plain):
        raise ValueError(f"plain must be a tuple of distinct cameras of the slice ({sorted(frames)}), got {plain!r}")
    return Snapshot(
        frame_index,
        {camera: np.ascontiguousarray(frame[::SCALE, ::SCALE]) for camera, frame in frames.items()},
        {camera: [(t.row, t.col, t.score, t.flagged) for t in r.tiles] for camera, r in result.results.items()},
        {camera: [(d.label, d.score, d.x1, d.y1, d.x2, d.y2) for d in r.detections] for camera, r in result.results.items()},
        [dict(record.to_dict(), slice=frame_index) for record in result.records],
        {full: frames[full].copy()} if full is not None else {},
        {full: list(result.boxes[full])} if full is not None and full in getattr(result, "boxes", {}) else {},
        {camera: frames[camera].copy() for camera in plain},
        _kept_maps(result, full))


def _kept_maps(result, full: Optional[str]) -> Dict[str, List[Tuple]]:
    if full is None or full not in result.results:
        return {}
    maps = [(t.row, t.col, t.anomaly_map.copy()) for t in result.results[full].tiles if t.anomaly_map is not None]
    return {full: maps} if maps else {}


def heat_colors(values: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """The heatmap's colour (float RGB, shape (..., 3)) and opacity (shape (...)) of each anomaly value, on the fixed
    scale set by the gate `threshold` (module docstring)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"the heatmap threshold must be a positive finite number, got {threshold!r}")
    values = np.asarray(values, np.float32)
    s = values / np.float32(threshold)
    above = values >= np.float32(threshold)
    t = np.clip((s - 1) / (HEAT_TOP - 1), 0, 1)[..., None]
    ramp = np.float32(HEAT_AT_COLOR) + t * (np.float32(HEAT_TOP_COLOR) - np.float32(HEAT_AT_COLOR))
    rgb = np.where(above[..., None], ramp, np.float32(HEAT_BELOW_COLOR)).astype(np.float32)
    fade = np.clip((s - HEAT_FLOOR) / (1 - HEAT_FLOOR), 0, 1) * HEAT_ALPHA_BELOW
    alpha = np.where(above, np.float32(HEAT_ALPHA_ABOVE), fade).astype(np.float32)
    return rgb, alpha


def heat_field(maps) -> np.ndarray:
    """The full-frame anomaly value of each pixel from [(row, col, 40x40 map)]: each cell over its 16x16 pixels, the
    higher value where the tile rows overlap, -inf where no tile was inspected."""
    out = np.full((FRAME_HEIGHT, FRAME_WIDTH), -np.inf, np.float32)
    origins = {(r, c): (x, y) for r, c, x, y in tile_origins()}
    cell = TILE_SIZE // MAP_SIZE
    for row, col, anomaly_map in maps:
        x, y = origins[(row, col)]
        cells = np.asarray(anomaly_map, np.float32).reshape(MAP_SIZE, MAP_SIZE)
        block = out[y:y + TILE_SIZE, x:x + TILE_SIZE]
        np.maximum(block, np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1), out=block)
    return out


def _blend_heat(picture: np.ndarray, maps, threshold: float) -> None:
    values = heat_field(maps)
    drawn = values / np.float32(threshold) > HEAT_FLOOR             # only the coloured pixels are worked on
    rgb, alpha = heat_colors(values[drawn], threshold)
    base = picture[drawn].astype(np.float32)
    picture[drawn] = np.rint(base + (rgb - base) * alpha[:, None]).astype(np.uint8)


def _shade_uninspected(region: np.ndarray, tiles: List[Tuple], scale: int = SCALE) -> None:
    inspected = {col for _, col, _, _ in tiles}
    width = TILE_SIZE // scale
    for col in range(TILE_COLS):
        if col not in inspected:
            part = region[:, col * width:(col + 1) * width]
            part[:] = (part.astype(np.float32) * SHADE).astype(np.uint8)


def render_annotated(snap: Snapshot) -> np.ndarray:
    """The 960x1080 picture of a slice: left camera above right, shaded uninspected columns, boxes with labels."""
    canvas = np.empty((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8)
    canvas[:] = BACKGROUND
    for index, camera in enumerate(CAMERAS):
        if camera in snap.frames:
            top = index * HALF_HEIGHT
            canvas[top:top + HALF_HEIGHT] = snap.frames[camera]
            if snap.tiles.get(camera):
                _shade_uninspected(canvas[top:top + HALF_HEIGHT], snap.tiles[camera])
    image = Image.fromarray(canvas)
    for index, camera in enumerate(CAMERAS):
        _draw_boxes(image, snap.detections.get(camera, ()), SCALE, index * HALF_HEIGHT)
    return np.asarray(image)


def _label(text: str, color, font) -> Image.Image:
    """The label strip at its base size: the text in black on the class colour, four pixels wider than the text."""
    width = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=font)) + 5
    strip = Image.new("RGB", (width, LABEL_HEIGHT), color)
    ImageDraw.Draw(strip).text((2, 1), text, fill=(0, 0, 0), font=font)
    return strip


def _draw_boxes(image: Image.Image, detections, scale: int, top: int, pad: int = 0, line: int = 1, label_scale: int = 1) -> None:
    """Boxes with label strips, for a camera drawn at 1/`scale` size with its top edge at row `top`. Each box is grown
    by `pad` px and outlined `line` px wide; its label is drawn `label_scale` times its base size."""
    width, height = FRAME_WIDTH // scale, FRAME_HEIGHT // scale
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    for detection in detections:                     # (label, score, x1, y1, x2, y2[, tag])
        label, score, x1, y1, x2, y2 = detection[:6]
        tag = detection[6] if len(detection) > 6 else ""
        color = CLASS_COLORS.get(label, _FALLBACK_COLOR)
        left = min(max(int(x1 / scale) - pad, 0), width - 1)
        right = min(max(int(x2 / scale) - 1 + pad, left), width - 1)
        upper = min(max(int(y1 / scale) - pad, 0), height - 1)
        lower = min(max(int(y2 / scale) - 1 + pad, upper), height - 1)
        draw.rectangle([left, top + upper, right, top + lower], outline=color, width=line)
        strip = _label(f"{label} {score:.2f}{tag}", color, font)
        if label_scale != 1:                         # whole pixels scaled up, so it looks the same on any Pillow
            strip = strip.resize((strip.width * label_scale, strip.height * label_scale), Image.NEAREST)
        strip_top = upper - strip.height if upper >= strip.height else upper
        image.paste(strip, (left, top + strip_top))


def render_camera(snap: Snapshot, camera: str, shade: bool = True, heat_threshold: Optional[float] = None) -> np.ndarray:
    """The 1920x1080 picture of one camera kept at full size in the snapshot: shaded uninspected columns, boxes, labels.
    With `heat_threshold` (the gate threshold) the snapshot's anomaly maps of that camera are blended in under the boxes."""
    if camera not in snap.full:
        raise ValueError(f"the snapshot keeps {sorted(snap.full)} at full size, not {camera!r}")
    picture = snap.full[camera].copy()
    if heat_threshold is not None and snap.heat.get(camera):
        _blend_heat(picture, snap.heat[camera], heat_threshold)
    if shade and snap.tiles.get(camera):
        _shade_uninspected(picture, snap.tiles[camera], scale=1)
    image = Image.fromarray(picture)
    if camera in snap.boxes:                         # the logged boxes; one found by the other camera is tagged (L) or (R)
        drawn = [(b.label, b.score, b.x1, b.y1, b.x2, b.y2, "" if b.source == camera else f" ({b.source[0].upper()})")
                 for b in snap.boxes[camera]]
    else:
        drawn = snap.detections.get(camera, ())
    _draw_boxes(image, drawn, 1, 0, pad=BOX_PAD, line=BOX_LINE, label_scale=LABEL_SCALE)
    return np.asarray(image)


def encode_jpeg(picture: np.ndarray, quality: int = 80) -> bytes:
    """The picture (the two-camera canvas or one full-size camera) as JPEG bytes."""
    shapes = ((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (FRAME_HEIGHT, FRAME_WIDTH, 3))
    if not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or picture.shape not in shapes:
        raise ValueError(f"the picture must be a {CANVAS_HEIGHT}x{CANVAS_WIDTH}x3 or {FRAME_HEIGHT}x{FRAME_WIDTH}x3 uint8 array")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError(f"quality must be an integer from 1 to 100, got {quality!r}")
    out = io.BytesIO()
    Image.fromarray(picture).save(out, format="JPEG", quality=quality, subsampling=0)   # 4:4:4: thin box lines keep their colour
    return out.getvalue()


def crop_defect(frame: np.ndarray, box) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """A square window of `frame` around `box` (x1, y1, x2, y2 in its pixels): CROP_MARGIN times the box's longer side,
    kept within CROP_MIN..CROP_MAX and inside the frame, thinned to at most CROP_SIZE a side. Returns the pixels (a
    copy) and the box in them. Only slicing, so it is cheap enough for the driver."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError(f"the frame must be a {FRAME_HEIGHT}x{FRAME_WIDTH}x3 uint8 array")
    if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        raise ValueError(f"the box must be four finite numbers, got {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (0 <= x1 < x2 <= FRAME_WIDTH and 0 <= y1 < y2 <= FRAME_HEIGHT):
        raise ValueError(f"the box must be non-empty and inside the frame, got {box!r}")
    side = int(min(max(round(CROP_MARGIN * max(x2 - x1, y2 - y1)), CROP_MIN), CROP_MAX))
    x0 = min(max(int(round((x1 + x2) / 2)) - side // 2, 0), FRAME_WIDTH - side)
    y0 = min(max(int(round((y1 + y2) / 2)) - side // 2, 0), FRAME_HEIGHT - side)
    step = -(-side // CROP_SIZE)
    pixels = frame[y0:y0 + side:step, x0:x0 + side:step].copy()
    return pixels, ((x1 - x0) / step, (y1 - y0) / step, (x2 - x0) / step, (y2 - y0) / step)


def crop_jpeg(pixels: np.ndarray, box, label: str, quality: int = 85) -> bytes:
    """A defect crop as JPEG bytes, with its box drawn in the class colour (the display process does this)."""
    if label not in CLASS_COLORS:
        raise ValueError(f"label must be one of {sorted(CLASS_COLORS)}, got {label!r}")
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 \
            or not 0 < max(pixels.shape[:2]) <= CROP_SIZE:
        raise ValueError(f"a crop must be an RGB uint8 array at most {CROP_SIZE} a side")
    image = Image.fromarray(pixels)
    x1, y1, x2, y2 = box
    ImageDraw.Draw(image).rectangle((round(x1), round(y1), round(x2) - 1, round(y2) - 1), outline=CLASS_COLORS[label], width=2)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, subsampling=0)
    return out.getvalue()
