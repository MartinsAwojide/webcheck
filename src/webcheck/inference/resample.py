# SPDX-FileCopyrightText: 2026 Martins Awojide
# SPDX-FileCopyrightText: Copyright © 1997-2011 by Secret Labs AB
# SPDX-FileCopyrightText: Copyright © 1995-2011 by Fredrik Lundh and contributors
# SPDX-FileCopyrightText: Copyright © 2010 by Jeffrey 'Alex' Clark and contributors
# SPDX-License-Identifier: MIT AND MIT-CMU

"""Pillow's Lanczos resize for uint8 images, ported from Pillow's `Resample.c` so the GPU kernel can give the
same bytes as `Image.resize(size, Image.LANCZOS)`: per-axis coefficient tables in double, their 22-bit fixed
point form, then a horizontal pass, a round-and-clip to uint8, and a vertical pass. `make_plan` holds the tables
the CUDA kernel (resample.cu) reads; `resample_u8` runs the same plan in NumPy and is the reference the kernel is
checked against on the board."""
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

PRECISION_BITS = 32 - 8 - 2
SUPPORT = 3.0


def lanczos_filter(x: float) -> float:
    if -3.0 <= x < 3.0:
        return _sinc(x) * _sinc(x / 3)
    return 0.0


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    x = x * math.pi
    return math.sin(x) / x


def _check_size(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def precompute_coeffs(in_size: int, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """(bounds, kk) for one axis of a full-image resize: bounds[i] = (first input index, number of taps) and
    kk[i] the normalised weights, zero beyond the taps."""
    in_size, out_size = _check_size(in_size, "in_size"), _check_size(out_size, "out_size")
    scale = filterscale = in_size / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = SUPPORT * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    inv_filterscale = 1.0 / filterscale
    bounds = np.zeros((out_size, 2), np.int32)
    kk = np.zeros((out_size, ksize), np.float64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        ww = 0.0
        for x in range(xmax):
            w = lanczos_filter((x + xmin - center + 0.5) * inv_filterscale)
            kk[xx, x] = w
            ww += w
        if ww != 0.0:
            for x in range(xmax):
                kk[xx, x] /= ww
        bounds[xx] = (xmin, xmax)
    return bounds, kk


def normalize_coeffs(kk: np.ndarray) -> np.ndarray:
    """The 22-bit fixed-point weights: round half away from zero, as Pillow's `(int)(+-0.5 + k * 2^22)`."""
    scaled = kk * float(1 << PRECISION_BITS)
    return np.where(kk < 0, np.trunc(scaled - 0.5), np.trunc(scaled + 0.5)).astype(np.int32)


@dataclass(frozen=True)
class ResamplePlan:
    in_width: int
    in_height: int
    out_width: int
    out_height: int
    need_horizontal: bool
    need_vertical: bool
    ksize_horizontal: int
    ksize_vertical: int
    horizontal_bounds: np.ndarray
    horizontal_kk: np.ndarray
    vertical_bounds: np.ndarray       # row indices relative to the first row the horizontal pass produces
    vertical_kk: np.ndarray
    first_row: int                    # first input row the horizontal pass reads
    rows: int                         # number of rows the horizontal pass produces


def make_plan(in_width: int, in_height: int, out_width: int, out_height: int) -> ResamplePlan:
    in_width, in_height = _check_size(in_width, "in_width"), _check_size(in_height, "in_height")
    out_width, out_height = _check_size(out_width, "out_width"), _check_size(out_height, "out_height")
    need_horizontal, need_vertical = out_width != in_width, out_height != in_height
    h_bounds, h_kk = precompute_coeffs(in_width, out_width)
    v_bounds, v_kk = precompute_coeffs(in_height, out_height)
    first_row, rows = 0, in_height
    if need_horizontal:
        first_row = int(v_bounds[0, 0])
        rows = int(v_bounds[-1, 0] + v_bounds[-1, 1]) - first_row
        v_bounds = v_bounds.copy()
        v_bounds[:, 0] -= first_row
    return ResamplePlan(in_width, in_height, out_width, out_height, need_horizontal, need_vertical,
                        h_kk.shape[1], v_kk.shape[1], h_bounds, normalize_coeffs(h_kk), v_bounds, normalize_coeffs(v_kk),
                        first_row, rows)


def _clip8(acc: np.ndarray) -> np.ndarray:
    return np.clip(acc >> PRECISION_BITS, 0, 255).astype(np.uint8)


def _pass(src: np.ndarray, axis: int, bounds: np.ndarray, kk: np.ndarray, ksize: int) -> np.ndarray:
    """One integer pass along `axis` (1 = columns, 0 = rows) of an H x W x C uint8 array."""
    n_in = src.shape[axis]
    shape = list(src.shape)
    shape[axis] = bounds.shape[0]
    acc = np.full(shape, 1 << (PRECISION_BITS - 1), np.int64)
    weight_shape = [1, 1, 1]
    weight_shape[axis] = -1
    for k in range(ksize):
        idx = np.minimum(bounds[:, 0] + k, n_in - 1)
        acc += np.take(src, idx, axis=axis).astype(np.int64) * kk[:, k].astype(np.int64).reshape(weight_shape)
    return _clip8(acc)


def _check_image(image) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3) or 0 in image.shape:
        raise ValueError(f"image must be H x W or H x W x 3 and not empty, got {image.shape}")


def _check_out_size(size) -> Tuple[int, int]:
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"size must be (width, height), got {size!r}")
    return _check_size(size[0], "width"), _check_size(size[1], "height")


def resample_u8(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """The Lanczos resize of an H x W (or H x W x 3) uint8 image to size = (width, height); equal to
    `np.asarray(Image.fromarray(image).resize(size, Image.LANCZOS))`."""
    _check_image(image)
    out_width, out_height = _check_out_size(size)
    flat = image.ndim == 2
    src = image[:, :, None] if flat else image
    plan = make_plan(src.shape[1], src.shape[0], out_width, out_height)
    tmp = src
    if plan.need_horizontal:
        tmp = _pass(src[plan.first_row:plan.first_row + plan.rows], 1, plan.horizontal_bounds, plan.horizontal_kk,
                    plan.ksize_horizontal)
    if plan.need_vertical:
        tmp = _pass(tmp, 0, plan.vertical_bounds, plan.vertical_kk, plan.ksize_vertical)
    out = np.array(tmp, copy=True)
    return out[:, :, 0] if flat else out
