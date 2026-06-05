"""Generates edge PNGs for pixdiff vs rs/src/bin/test_edge.rs."""

import numpy as np
from PIL import Image
from edge import edge

PPI = (600.0, 600.0)
S   = 64
BIG = 3500


def write_png(rgba_f32: np.ndarray, path: str) -> None:
    """Direct u8 encode: 0.0->0, 1.0->255 (black/white/transparent only)."""
    out = (rgba_f32 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(path, dpi=PPI)
    edge_px = int((rgba_f32[..., 3] > 0.5).sum())
    print(f"wrote {path}  edge_px={edge_px}")


def solid(h: int, w: int, rgb: tuple[float, float, float]) -> np.ndarray:
    buf = np.zeros((h, w, 4), dtype=np.float32)
    buf[:, :, 0] = np.float32(rgb[0])
    buf[:, :, 1] = np.float32(rgb[1])
    buf[:, :, 2] = np.float32(rgb[2])
    buf[:, :, 3] = np.float32(1.0)
    return buf


def split(size: int) -> np.ndarray:
    """Left half one color, right half another."""
    buf = solid(size, size, (0.2, 0.4, 0.6))
    buf[:, size//2:, 0] = np.float32(0.8)
    buf[:, size//2:, 1] = np.float32(0.2)
    buf[:, size//2:, 2] = np.float32(0.3)
    return buf


def quad(size: int) -> np.ndarray:
    """Four quadrant blocks."""
    buf = np.zeros((size, size, 4), dtype=np.float32)
    h2, w2 = size // 2, size // 2
    buf[:h2, :w2] = np.float32([0.1, 0.9, 0.1, 1.0])
    buf[:h2, w2:] = np.float32([0.9, 0.1, 0.1, 1.0])
    buf[h2:, :w2] = np.float32([0.1, 0.1, 0.9, 1.0])
    buf[h2:, w2:] = np.float32([0.9, 0.9, 0.9, 1.0])
    return buf


def plasma_crop(size: int) -> np.ndarray:
    """Top-left crop of plasma (seed=0, turb=1.0), linear f32."""
    from plasma import diamond_square, RNG
    SEED_R = 0
    SEED_G = 0 ^ 0x9E3779B97F4A7C15
    SEED_B = 0 ^ 0xD1B54A32D192ED03
    r = diamond_square(RNG(SEED_R), 1.0)[:size, :size].astype(np.float32)
    g = diamond_square(RNG(SEED_G), 1.0)[:size, :size].astype(np.float32)
    b = diamond_square(RNG(SEED_B), 1.0)[:size, :size].astype(np.float32)
    a = np.ones((size, size), dtype=np.float32)
    return np.stack([r, g, b, a], axis=-1)


def gradient(size: int) -> np.ndarray:
    """R=col/(size-1), G=row/(size-1), B=0.5, A=1."""
    buf = np.zeros((size, size, 4), dtype=np.float32)
    cols = (np.arange(size, dtype=np.float32) / np.float32(size - 1))
    rows = (np.arange(size, dtype=np.float32) / np.float32(size - 1))
    buf[:, :, 0] = cols[np.newaxis, :]
    buf[:, :, 1] = rows[:, np.newaxis]
    buf[:, :, 2] = np.float32(0.5)
    buf[:, :, 3] = np.float32(1.0)
    return buf


# ---- 64×64 cases ----
write_png(edge(split(S), 50.0, 1), "test_edge_split_a50_w1.png")
write_png(edge(split(S), 50.0, 3), "test_edge_split_a50_w3.png")
write_png(edge(split(S), 50.0, 7), "test_edge_split_a50_w7.png")

write_png(edge(quad(S), 50.0,  3), "test_edge_quad_a50_w3.png")
write_png(edge(quad(S),  0.0,  3), "test_edge_quad_a0_w3.png")
write_png(edge(quad(S), 100.0, 3), "test_edge_quad_a100_w3.png")

import sys
print("generating plasma crop...", file=sys.stderr)
write_png(edge(plasma_crop(S), 50.0, 3), "test_edge_plasma_a50_w3.png")

print("generating 3500×3500 confirming run...", file=sys.stderr)
write_png(edge(gradient(BIG), 50.0, 3), "test_edge_confirm3500.png")
