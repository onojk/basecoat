"""Generates test PNGs for pixdiff vs rs/src/bin/test_punch.rs."""

import sys
import numpy as np
from PIL import Image
from punch import punch
from layers import linear_to_srgb

S   = 64
BIG = 3500
PPI = (600.0, 600.0)


def write_png(rgba_f32: np.ndarray, path: str) -> None:
    """Encode linear f32 RGBA → sRGB u8 PNG (same path as test_layers.py)."""
    lin = rgba_f32.astype(np.float64)
    r8  = (linear_to_srgb(lin[..., 0]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    g8  = (linear_to_srgb(lin[..., 1]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    b8  = (linear_to_srgb(lin[..., 2]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    a8  = (lin[..., 3].clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(np.stack([r8, g8, b8, a8], axis=-1), "RGBA").save(path, dpi=PPI)
    print(f"wrote {path}")


def gradient(size: int) -> np.ndarray:
    """Linear gradient: R = col/(size-1), G = row/(size-1), B = 0.5, A = 1.0."""
    buf = np.zeros((size, size, 4), dtype=np.float32)
    cols = (np.arange(size, dtype=np.float32) / np.float32(size - 1))
    rows = (np.arange(size, dtype=np.float32) / np.float32(size - 1))
    buf[:, :, 0] = cols[np.newaxis, :]
    buf[:, :, 1] = rows[:, np.newaxis]
    buf[:, :, 2] = np.float32(0.5)
    buf[:, :, 3] = np.float32(1.0)
    return buf


def plasma_buf(size: int) -> np.ndarray:
    """Top-left crop of plasma (seed=0, turb=1.0) — reuse plasma.py logic."""
    from plasma import diamond_square, RNG, linear_to_srgb as p_srgb
    SEED_R = 0
    SEED_G = 0 ^ 0x9E3779B97F4A7C15
    SEED_B = 0 ^ 0xD1B54A32D192ED03
    r = diamond_square(RNG(SEED_R), 1.0)[:size, :size].astype(np.float32)
    g = diamond_square(RNG(SEED_G), 1.0)[:size, :size].astype(np.float32)
    b = diamond_square(RNG(SEED_B), 1.0)[:size, :size].astype(np.float32)
    a = np.ones((size, size), dtype=np.float32)
    return np.stack([r, g, b, a], axis=-1)


# ---- test cases ----

def run_case(buf: np.ndarray, k: float, sat: float, passes: int, name: str) -> None:
    out = punch(buf.copy(), k, sat, passes)
    write_png(out, name)


grad = gradient(S)

run_case(grad, 9.0, 4.0, 6, "test_punch_grad_default.png")
run_case(grad, 0.0, 1.0, 1, "test_punch_grad_identity.png")
run_case(grad, 9.0, 0.0, 1, "test_punch_grad_grayscale.png")
run_case(grad, 9.0, 4.0, 1, "test_punch_grad_1pass.png")

print("generating plasma crop...", file=sys.stderr)
plasma = plasma_buf(S)
run_case(plasma, 9.0, 4.0, 6, "test_punch_plasma_default.png")

# Confirming run at 3500x3500
print("generating 3500x3500 confirming run...", file=sys.stderr)
big = gradient(BIG)
run_case(big, 9.0, 4.0, 6, "test_punch_3500.png")
