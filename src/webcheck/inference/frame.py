"""Cutting a 1920x1080 RGB frame into the six 640x640 model-input tiles (two rows, three columns,
row-major; the rows overlap)."""
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TILE_SIZE = 640
TILE_ROWS = 2
TILE_COLS = 3


class FrameError(ValueError):
    """A frame or tile that is not the shape or type the models expect."""


def tile_columns(columns: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    """The tile columns to inspect, sorted; None means all of them."""
    if columns is None:
        return tuple(range(TILE_COLS))
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Iterable):
        raise ValueError(f"columns must be a sequence of tile column numbers, got {columns!r}")
    columns = list(columns)
    if not columns or any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < TILE_COLS for c in columns) \
            or len(set(columns)) != len(columns):
        raise ValueError(f"columns must be distinct integers from 0 to {TILE_COLS - 1}, got {columns!r}")
    return tuple(sorted(columns))


def tile_origins(columns: Optional[Sequence[int]] = None) -> List[Tuple[int, int, int, int]]:
    """(row, col, x, y) of each tile's top-left corner, row-major, for the given tile columns (default all)."""
    row_step = FRAME_HEIGHT - TILE_SIZE
    return [(r, c, c * TILE_SIZE, r * row_step) for r in range(TILE_ROWS) for c in tile_columns(columns)]


def tile_views(frame, columns: Optional[Sequence[int]] = None) -> List[Tuple[int, int, np.ndarray]]:
    """(row, col, tile) for each tile of the given columns (default all six), as views of `frame` (no copies); the
    frame must not change while they are in use."""
    if not isinstance(frame, np.ndarray):
        raise FrameError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise FrameError(f"frame must be uint8, got {frame.dtype}")
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise FrameError(f"frame must be {FRAME_HEIGHT}x{FRAME_WIDTH}x3 (HxWxRGB), got {frame.shape}")
    return [(r, c, frame[y:y + TILE_SIZE, x:x + TILE_SIZE]) for r, c, x, y in tile_origins(columns)]


def crop_tiles(frame) -> List[Tuple[int, int, np.ndarray]]:
    """(row, col, tile) for each of the six tiles; tiles are copies, so the frame buffer can be reused."""
    return [(r, c, tile.copy()) for r, c, tile in tile_views(frame)]
