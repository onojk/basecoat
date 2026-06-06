"""Dihedral kaleidoscope transform — see spec/kaleido.md.
Mirrors rs/src/kaleido.rs exactly.
"""

from __future__ import annotations
import math
import numpy as np
from sample import sample_bilinear


def kaleido(
    source: np.ndarray,
    segments: int,
    rotation_deg: float,
    zoom: float,
) -> np.ndarray:
    """Apply dihedral kaleidoscope to source.

    source       : (H, W, 4) float32 linear RGBA
    segments     : number of dihedral wedges (2–24)
    rotation_deg : pattern rotation in degrees (0–360)
    zoom         : >1 samples a smaller central source region (0.25–4.0)
    Returns      : (H, W, 4) float32 linear RGBA (new array)
    """
    h, w = source.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    rotation = rotation_deg * math.pi / 180.0
    wedge_w = 2.0 * math.pi / segments

    # Build f64 coordinate grids
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)      # (H, W)

    dx = xx - cx
    dy = yy - cy
    r  = np.hypot(dx, dy)
    a  = np.arctan2(dy, dx) + rotation   # float64

    # --- Dihedral fold ---
    k_f = np.floor(a / wedge_w)          # float64, integer values
    am  = a - k_f * wedge_w              # residual in [0, wedge_w]
    am  = np.clip(am, 0.0, wedge_w)      # fp guard at exact multiples

    # Odd wedge → mirror; use int64 for correct negative-k modulo
    k_int  = k_f.astype(np.int64)
    mirror = (k_int % 2) == 1
    am     = np.where(mirror, wedge_w - am, am)

    # --- Source coordinates ---
    sx = cx + (r / zoom) * np.cos(am)
    sy = cy + (r / zoom) * np.sin(am)

    # --- Bilinear sample (edge-clamped, f64 arithmetic) ---
    sampled = sample_bilinear(source, sx, sy)   # (H, W, 4) float64
    return np.clip(sampled, 0.0, 1.0).astype(np.float32)
