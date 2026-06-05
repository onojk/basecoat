"""Layer stack + compositor engine — see spec/layers.md."""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

MAX_LAYERS = 25
UNDO_DEPTH = 20


class BlendMode(Enum):
    Normal     = "normal"
    Multiply   = "multiply"
    Screen     = "screen"
    Overlay    = "overlay"
    Difference = "difference"


# ---------------------------------------------------------------------------
# sRGB <-> linear  (IEC 61966-2-1, same as plasma.py)
# ---------------------------------------------------------------------------

def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92,
                    ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Blend modes (linear, per-channel, top=a over bottom=b)
# ---------------------------------------------------------------------------

def _blend(mode: BlendMode, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if mode is BlendMode.Normal:
        return a
    if mode is BlendMode.Multiply:
        return a * b
    if mode is BlendMode.Screen:
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode is BlendMode.Overlay:
        return np.where(b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    if mode is BlendMode.Difference:
        return np.abs(a - b)
    raise ValueError(f"Unknown blend mode {mode}")


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """RGBA buffer in linear-light straight alpha."""
    rgba: np.ndarray          # shape (H, W, 4), float32, [0,1]
    mode: BlendMode = BlendMode.Normal
    opacity: float = 1.0
    visible: bool = True
    name: str = ""

    def copy(self) -> Layer:
        return Layer(self.rgba.copy(), self.mode, self.opacity, self.visible, self.name)


def make_layer(height: int, width: int, color=(0.0, 0.0, 0.0, 0.0),
               mode=BlendMode.Normal, opacity=1.0, visible=True, name="") -> Layer:
    buf = np.full((height, width, 4), color, dtype=np.float32)
    return Layer(buf, mode, opacity, visible, name)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def composite(layers: list[Layer]) -> np.ndarray:
    """Flatten visible layers bottom->top; return (H,W,4) float32 linear."""
    if not layers:
        raise ValueError("empty layer list")
    H, W = layers[0].rgba.shape[:2]
    acc = np.zeros((H, W, 4), dtype=np.float64)  # (Cb_r, Cb_g, Cb_b, Ab)

    for layer in layers:
        if not layer.visible:
            continue
        src = layer.rgba.astype(np.float64)
        Cs = src[..., :3]
        As = src[..., 3:4] * float(layer.opacity)   # effective alpha

        Cb = acc[..., :3]
        Ab = acc[..., 3:4]

        blended = _blend(layer.mode, Cs, Cb)

        Aout = As + Ab * (1.0 - As)
        # Avoid div-by-zero; where Aout==0 result is 0
        safe = np.where(Aout > 0.0, Aout, 1.0)
        Cout = (blended * As + Cb * Ab * (1.0 - As)) / safe
        Cout = np.where(Aout > 0.0, Cout, 0.0)

        acc[..., :3] = Cout
        acc[..., 3:4] = Aout

    return np.clip(acc, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Undo snapshot helpers
# ---------------------------------------------------------------------------

@dataclass
class _PixelSnap:
    layer_idx: int
    buf: np.ndarray

@dataclass
class _StructSnap:
    layers: list[Layer]   # deep copies


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

class Stack:
    def __init__(self):
        self.layers: list[Layer] = []
        self._undo: list[_PixelSnap | _StructSnap] = []

    # -- undo ring --

    def _push_pixel_snap(self, idx: int):
        snap = _PixelSnap(idx, self.layers[idx].rgba.copy())
        self._undo.append(snap)
        if len(self._undo) > UNDO_DEPTH:
            self._undo.pop(0)

    def _push_struct_snap(self):
        snap = _StructSnap([l.copy() for l in self.layers])
        self._undo.append(snap)
        if len(self._undo) > UNDO_DEPTH:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        snap = self._undo.pop()
        if isinstance(snap, _PixelSnap):
            self.layers[snap.layer_idx].rgba = snap.buf
        else:
            self.layers = [l.copy() for l in snap.layers]
        return True

    # -- structural ops --

    def add(self, layer: Layer) -> Optional[str]:
        """Returns error string if at cap, else None."""
        if len(self.layers) >= MAX_LAYERS:
            return f"MAX_LAYERS ({MAX_LAYERS}) reached"
        self._push_struct_snap()
        self.layers.append(layer)
        return None

    def remove(self, idx: int):
        self._push_struct_snap()
        self.layers.pop(idx)

    def reorder(self, from_idx: int, to_idx: int):
        self._push_struct_snap()
        layer = self.layers.pop(from_idx)
        self.layers.insert(to_idx, layer)

    def flatten_visible(self):
        """Composite visible layers into one; preserve invisible in place."""
        self._push_struct_snap()
        visible_indices = [i for i, l in enumerate(self.layers) if l.visible]
        if not visible_indices:
            return
        merged_buf = composite(self.layers)
        merged = Layer(merged_buf, BlendMode.Normal, 1.0, True, "merged")
        insert_at = visible_indices[0]
        # Remove visible layers (iterate in reverse to preserve indices)
        for i in reversed(visible_indices):
            self.layers.pop(i)
        self.layers.insert(insert_at, merged)

    # -- pixel ops --

    def fill(self, idx: int, color: tuple[float, float, float, float]):
        self._push_pixel_snap(idx)
        self.layers[idx].rgba[...] = color

    def composite(self) -> np.ndarray:
        return composite(self.layers)
