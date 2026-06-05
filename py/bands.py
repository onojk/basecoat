"""Band generator — concentric outward fill from a seed mask.
See spec/bands.md.  Mirrors rs/src/bands.rs exactly for delta-0 diffs.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Band:
    mask: np.ndarray   # (H, W) bool
    white: bool        # True = white, False = black


def growth_to_thickness(growth_pct: float) -> int:
    """Map growth% [0,100] -> thickness_px [1,40] with round-half-up."""
    x = 1.0 + (growth_pct / 100.0) * 39.0
    return int(x + 0.5)  # floor(x + 0.5) = round-half-up (matches Rust)


def dilate_chebyshev(mask: np.ndarray, r: int) -> np.ndarray:
    """Chebyshev (box) dilation by r, two-pass 1D sliding window.
    Matches rs/src/bands.rs::dilate_chebyshev exactly.
    mask: (H, W) bool ndarray.
    """
    if r == 0:
        return mask.copy()
    h, w = mask.shape

    # ---- Horizontal pass ----
    tmp = np.zeros((h, w), dtype=bool)
    for x in range(w):
        x_lo = max(0, x - r)
        x_hi = min(w - 1, x + r)
        tmp[:, x] = np.any(mask[:, x_lo:x_hi + 1], axis=1)

    # ---- Vertical pass ----
    result = np.zeros((h, w), dtype=bool)
    for y in range(h):
        y_lo = max(0, y - r)
        y_hi = min(h - 1, y + r)
        result[y, :] = np.any(tmp[y_lo:y_hi + 1, :], axis=0)

    return result


def generate_bands(
    seed: np.ndarray,      # (H, W) bool — opaque pixels of seed layer
    thickness_px: int,
    fill_pct: float,       # 0..100
) -> list[Band]:
    """Generate bands outward from seed, innermost first (index 0 = white)."""
    h, w = seed.shape
    total = w * h
    coverage = seed.copy()
    coverage_count = int(coverage.sum())
    fill_threshold = int(np.ceil(fill_pct / 100.0 * total))
    bands: list[Band] = []

    while True:
        dilated  = dilate_chebyshev(coverage, thickness_px)
        new_ring = dilated & ~coverage

        new_count = int(new_ring.sum())
        if new_count == 0:
            break  # stop condition 2: canvas saturated

        is_white = len(bands) % 2 == 0  # white first
        bands.append(Band(mask=new_ring.copy(), white=is_white))

        coverage      |= new_ring
        coverage_count += new_count

        if coverage_count >= fill_threshold:
            break  # stop condition 1: fill% reached

    return bands
