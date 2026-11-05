"""Tile to model input: NHWC float32 in [0, 1]. The gate sees a 320x320 Lanczos resize of the tile, the
detector sees the tile as is."""
import numpy as np
from PIL import Image

from webcheck.inference.frame import TILE_SIZE, FrameError

GATE_SIZE = 320


def _check_tile(tile) -> None:
    if not isinstance(tile, np.ndarray) or tile.dtype != np.uint8 or tile.shape != (TILE_SIZE, TILE_SIZE, 3):
        shape = getattr(tile, "shape", None)
        dtype = getattr(tile, "dtype", type(tile).__name__)
        raise FrameError(f"tile must be a {TILE_SIZE}x{TILE_SIZE}x3 uint8 array, got {shape} {dtype}")


def gate_input(tile: np.ndarray) -> np.ndarray:
    _check_tile(tile)
    small = Image.fromarray(tile).resize((GATE_SIZE, GATE_SIZE), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(small, dtype=np.float32)[None] / 255.0)


def detector_input(tile: np.ndarray) -> np.ndarray:
    _check_tile(tile)
    return np.ascontiguousarray(tile.astype(np.float32)[None] / 255.0)
