"""Generates test PNGs matching rs/src/bin/test_layers.rs exactly."""

import sys
import numpy as np
from PIL import Image
from layers import (
    Layer, BlendMode, Stack, composite, make_layer,
    linear_to_srgb,
)

PPM_DPI = (600.0, 600.0)


def write_png(layer: Layer, path: str):
    h, w = layer.rgba.shape[:2]
    lin = layer.rgba.astype(np.float64)
    r8 = (linear_to_srgb(lin[..., 0]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    g8 = (linear_to_srgb(lin[..., 1]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    b8 = (linear_to_srgb(lin[..., 2]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    a8 = (lin[..., 3].clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    rgba = np.stack([r8, g8, b8, a8], axis=-1)
    Image.fromarray(rgba, "RGBA").save(path, dpi=PPM_DPI)
    print(f"wrote {path}")


S = 64


def make(color, mode=BlendMode.Normal, opacity=1.0, visible=True, size=S):
    buf = np.full((size, size, 4), color, dtype=np.float32)
    return Layer(buf, mode, opacity, visible)


# --- test_normal ---
bot = make([0.0, 0.0, 1.0, 1.0])
top = make([1.0, 0.0, 0.0, 1.0])
write_png(Layer(composite([bot, top])), "test_normal.png")

# --- test_multiply ---
bot = make([0.0, 0.0, 0.5, 1.0])
top = make([0.8, 0.0, 0.0, 1.0], BlendMode.Multiply)
write_png(Layer(composite([bot, top])), "test_multiply.png")

# --- test_screen ---
bot = make([0.0, 0.0, 0.5, 1.0])
top = make([0.8, 0.0, 0.0, 1.0], BlendMode.Screen)
write_png(Layer(composite([bot, top])), "test_screen.png")

# --- test_overlay ---
bot = make([0.0, 0.0, 0.3, 1.0])
top = make([0.8, 0.0, 0.0, 1.0], BlendMode.Overlay)
write_png(Layer(composite([bot, top])), "test_overlay.png")

# --- test_difference ---
bot = make([0.5, 0.5, 0.5, 1.0])
top = make([0.8, 0.2, 0.3, 1.0], BlendMode.Difference)
write_png(Layer(composite([bot, top])), "test_difference.png")

# --- test_opacity ---
bot = make([0.0, 0.8, 0.0, 1.0])
top = make([1.0, 0.0, 0.0, 1.0], opacity=0.5)
write_png(Layer(composite([bot, top])), "test_opacity.png")

# --- test_alpha ---
bot = make([0.0, 0.0, 1.0, 0.8])
top = make([1.0, 0.0, 0.0, 0.6])
write_png(Layer(composite([bot, top])), "test_alpha.png")

# --- test_invisible ---
bot = make([0.0, 0.5, 0.0, 1.0])
mid = make([1.0, 0.0, 0.0, 1.0], visible=False)
top = make([0.0, 0.0, 0.8, 0.5])
write_png(Layer(composite([bot, mid, top])), "test_invisible.png")

# --- test_flatten ---
stack = Stack()
a = make([0.2, 0.4, 0.6, 1.0])
b = make([0.8, 0.1, 0.1, 0.7], BlendMode.Screen)
stack.add(a)
stack.add(b)
stack.flatten_visible()
result = stack.composite()
write_png(Layer(result), "test_flatten.png")

# --- test_undo ---
stack = Stack()
a = make([0.3, 0.3, 0.3, 1.0])
stack.add(a)
stack.fill(0, (1.0, 0.0, 0.0, 1.0))
stack.undo()
result = stack.composite()
write_png(Layer(result), "test_undo.png")

# --- test_3500 (full-size confirming run) ---
BIG = 3500
bot = make([0.18, 0.18, 0.18, 1.0], size=BIG)
top = make([0.8, 0.4, 0.1, 0.5], BlendMode.Overlay, size=BIG)
result = composite([bot, top])
write_png(Layer(result), "test_3500.png")
