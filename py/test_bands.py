"""Generates composite PNGs of band generation for pixdiff vs test_bands.rs."""

import numpy as np
from PIL import Image
from bands import generate_bands, growth_to_thickness

PPI = (600.0, 600.0)


def make_composite(seed: np.ndarray, bands_list, path: str) -> None:
    """Composite seed + bands into one RGBA PNG.
    seed pixels → black opaque; band-k pixels → white/black opaque; else transparent.
    Black/white in linear space = sRGB 0/255 exactly; no floating point needed.
    """
    h, w = seed.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Seed: black opaque
    rgba[seed, :] = [0, 0, 0, 255]
    # Bands: innermost first = drawn last (but non-overlapping so order doesn't matter)
    for band in bands_list:
        v = 255 if band.white else 0
        rgba[band.mask, :3] = v
        rgba[band.mask, 3]  = 255
    Image.fromarray(rgba, "RGBA").save(path, dpi=PPI)
    print(f"wrote {path}  bands={len(bands_list)}")


def dot_seed(size: int) -> np.ndarray:
    s = np.zeros((size, size), dtype=bool)
    s[size // 2, size // 2] = True
    return s


def diag_seed(size: int) -> np.ndarray:
    s = np.zeros((size, size), dtype=bool)
    for i in range(size):
        s[i, i] = True
    return s


S = 64
C = 256  # confirming run

# dot, growth=30 (thickness=13)
t = growth_to_thickness(30)
seed = dot_seed(S)
bs   = generate_bands(seed, t, 90.0)
make_composite(seed, bs, "test_bands_dot_g30.png")

# dot, growth=10 (thickness=5)
t    = growth_to_thickness(10)
bs   = generate_bands(seed, t, 90.0)
make_composite(seed, bs, "test_bands_dot_g10.png")

# dot, growth=50 (thickness=20)
t    = growth_to_thickness(50)
bs   = generate_bands(seed, t, 90.0)
make_composite(seed, bs, "test_bands_dot_g50.png")

# diagonal line, growth=30
t    = growth_to_thickness(30)
seed = diag_seed(S)
bs   = generate_bands(seed, t, 90.0)
make_composite(seed, bs, "test_bands_diag_g30.png")

# confirming run 256x256 dot, growth=30
t    = growth_to_thickness(30)
seed = dot_seed(C)
bs   = generate_bands(seed, t, 80.0)
make_composite(seed, bs, "test_bands_confirm256.png")
