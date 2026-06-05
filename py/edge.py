"""Edge technique — posterize-then-outline. See spec/edge.md.
Mirrors rs/src/edge.rs exactly. Reuses dilate_chebyshev from bands.py.
"""

from __future__ import annotations
import argparse
import math
import numpy as np
from PIL import Image
from bands import dilate_chebyshev


def aggr_to_n(aggr: float) -> int:
    """Map aggressiveness [0,100] -> posterize level count N [3,32] (round-half-up)."""
    return int(math.floor(3.0 + (aggr / 100.0) * 29.0 + 0.5))


def posterize_buckets(rgb_f32: np.ndarray, n: int) -> np.ndarray:
    """(H, W, 3) float32 -> (H, W, 3) int32 bucket indices [0..n-1], round-half-up.
    Promotes to float64 before multiply to match Rust's `c as f64 * (n-1)` path.
    """
    rgb64   = rgb_f32.astype(np.float64)
    buckets = np.floor(rgb64 * (n - 1) + 0.5).astype(np.int32)
    return np.clip(buckets, 0, n - 1)


def edge(rgba: np.ndarray, aggr: float, line_width: int) -> np.ndarray:
    """Compute edge layer for source rgba (H,W,4) float32.

    Returns a new (H,W,4) float32 array:
    edge pixels = (0,0,0,1) opaque black; others = (0,0,0,0) transparent.
    Source array is not modified.
    """
    h, w = rgba.shape[:2]
    n = aggr_to_n(aggr)

    # Step 1 — posterize
    bucks = posterize_buckets(rgba[..., :3], n)   # (H, W, 3) int32

    # Step 2 — 4-neighbour boundary detection (no wrap; border = same)
    # Pixel (y,x) is edge if any 4-neighbour has a different triple.
    diff_vert  = np.any(bucks[:-1, :] != bucks[1:,  :], axis=2)  # (H-1, W)
    diff_horiz = np.any(bucks[:, :-1] != bucks[:, 1:],  axis=2)  # (H, W-1)

    mask = np.zeros((h, w), dtype=bool)
    mask[:-1, :] |= diff_vert    # row y   differs from row y+1
    mask[1:,  :] |= diff_vert    # row y+1 differs from row y
    mask[:, :-1] |= diff_horiz   # col x   differs from col x+1
    mask[:, 1:]  |= diff_horiz   # col x+1 differs from col x

    # Step 3 — dilation (reuse Chebyshev from bands)
    r = line_width // 2
    if r > 0:
        mask = dilate_chebyshev(mask, r)

    # Step 4 — build output layer
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[mask, 3] = 1.0   # alpha = 1; RGB stays 0 = opaque black

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--aggr",  type=float, default=50.0)
    ap.add_argument("--width", type=int,   default=3)
    args = ap.parse_args()

    img  = Image.open(args.input).convert("RGBA")
    rgba = np.array(img, dtype=np.float32) / 255.0
    out  = edge(rgba, args.aggr, args.width)
    result = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(result, "RGBA").save(args.output)
    print(f"Wrote {args.output}  N={aggr_to_n(args.aggr)}  r={args.width // 2}")


if __name__ == "__main__":
    main()
