"""Punch adjustment — contrast S-curve + saturation, N passes.
See spec/punch.md.  Float model: compute in f64, store as f32.
"""

from __future__ import annotations
import argparse
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def punch(rgba: np.ndarray, k: float, sat: float, passes: int) -> np.ndarray:
    """Apply punch in-place to a (H, W, 4) float32 RGBA array.

    Computation done in f64 per pass; result clamped and stored back as f32.
    Alpha channel is untouched.
    Returns the modified array (same object).
    """
    k64   = np.float64(k)
    sat64 = np.float64(sat)

    # Pre-compute contrast normalisation constants (constant for a given k)
    if k64 >= np.float64(1e-6):
        def _sig(x: np.ndarray) -> np.ndarray:
            return np.float64(1.0) / (np.float64(1.0) + np.exp(-k64 * (x - np.float64(0.5))))
        lo         = _sig(np.float64(0.0))
        hi_minus_lo = _sig(np.float64(1.0)) - lo

    for _ in range(passes):
        # Read RGB as f64
        rgb = rgba[..., :3].astype(np.float64)

        # --- Contrast ---
        if k64 < np.float64(1e-6):
            c = rgb
        else:
            s = np.float64(1.0) / (np.float64(1.0) + np.exp(-k64 * (rgb - np.float64(0.5))))
            c = (s - lo) / hi_minus_lo

        # --- Saturation (luma from post-contrast RGB) ---
        luma = (np.float64(0.2126) * c[..., 0]
              + np.float64(0.7152) * c[..., 1]
              + np.float64(0.0722) * c[..., 2])[..., np.newaxis]
        sat_rgb = np.clip(luma + sat64 * (c - luma), 0.0, 1.0)

        # Write back as f32
        rgba[..., :3] = sat_rgb.astype(np.float32)

    return rgba


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Apply punch to a PNG.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--contrast",   type=float, default=9.0)
    ap.add_argument("--saturation", type=float, default=4.0)
    ap.add_argument("--passes",     type=int,   default=6)
    args = ap.parse_args()

    img  = Image.open(args.input).convert("RGBA")
    rgba = np.array(img, dtype=np.float32) / 255.0
    punch(rgba, args.contrast, args.saturation, args.passes)
    out  = (rgba * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
