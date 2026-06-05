#!/usr/bin/env python3
"""
pixdiff.py  <image-a> <image-b> [--tolerance N]

Compare two images pixel-by-pixel.
Exits 0 when max per-channel delta <= tolerance (default 0).
Exits 1 on dimension/mode mismatch or when delta exceeds tolerance.
"""

import argparse
import sys
import numpy as np
from PIL import Image


def load(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.int16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="max allowed per-channel delta (default 0)")
    args = ap.parse_args()

    img_a = Image.open(args.a)
    img_b = Image.open(args.b)

    if img_a.size != img_b.size:
        print(f"FAIL  size mismatch: {img_a.size} vs {img_b.size}", file=sys.stderr)
        sys.exit(1)

    arr_a = np.array(img_a.convert("RGBA"), dtype=np.int16)
    arr_b = np.array(img_b.convert("RGBA"), dtype=np.int16)

    delta = np.abs(arr_a - arr_b)
    max_delta = int(delta.max())
    differing = int(np.any(delta > 0, axis=-1).sum())

    print(f"size        : {img_a.size[0]}x{img_a.size[1]}")
    print(f"max delta   : {max_delta}")
    print(f"diff pixels : {differing}")
    print(f"tolerance   : {args.tolerance}")

    if max_delta > args.tolerance:
        print("RESULT: FAIL", file=sys.stderr)
        sys.exit(1)

    print("RESULT: PASS")


if __name__ == "__main__":
    main()
