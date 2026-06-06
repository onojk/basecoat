"""Shared bilinear sampler — mirrors rs/src/sample.rs exactly.

Future transforms (rotate, scale, mirror) should reuse sample_bilinear
rather than re-implementing interpolation.
"""

from __future__ import annotations
import numpy as np


def sample_bilinear(src: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Bilinear-sample src at float coordinates (sx, sy), edge-clamped.

    src : (H, W, C) float32
    sx  : (H, W) float64, x pixel-space coords (0 = left column)
    sy  : (H, W) float64, y pixel-space coords (0 = top row)
    Returns (H, W, C) float64.

    Out-of-bounds coordinates are clamped to the image edge (no transparent
    fringe).  All arithmetic is f64.
    """
    h, w = src.shape[:2]
    src64 = src.astype(np.float64)

    sx = np.clip(sx, 0.0, w - 1)
    sy = np.clip(sy, 0.0, h - 1)

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    tx = sx - x0.astype(np.float64)
    ty = sy - y0.astype(np.float64)

    w00 = ((1.0 - tx) * (1.0 - ty))[..., np.newaxis]
    w10 = (tx          * (1.0 - ty))[..., np.newaxis]
    w01 = ((1.0 - tx) * ty          )[..., np.newaxis]
    w11 = (tx          * ty          )[..., np.newaxis]

    return (w00 * src64[y0, x0] +
            w10 * src64[y0, x1] +
            w01 * src64[y1, x0] +
            w11 * src64[y1, x1])
