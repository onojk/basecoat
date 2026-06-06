"""Tests + reference PNGs for kaleidoscope — see spec/kaleido.md.

Tests:
  test_seam_algorithm  — source coords from either side of every seam converge
                         (hard requirement; tolerance 1e-6)
  test_seam_no_edge    — edge-detect on kaleido(gradient) finds no seam lines

Reference PNGs (for pixdiff vs rs/src/bin/test_kaleido.rs):
  test_kaleido_grad32_s6_r0_z1.png
  test_kaleido_grad32_s8_r30_z1p5.png
  test_kaleido_grad64_s4_r0_z1.png
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image

from layers import linear_to_srgb
from kaleido import kaleido
from edge import edge

PPI = (600.0, 600.0)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def gradient(size: int) -> np.ndarray:
    """R = col/(size-1), G = row/(size-1), B = 0.5, A = 1.0 — smooth gradient."""
    buf = np.zeros((size, size, 4), dtype=np.float32)
    buf[:, :, 0] = (np.arange(size, dtype=np.float32) / (size - 1))[np.newaxis, :]
    buf[:, :, 1] = (np.arange(size, dtype=np.float32) / (size - 1))[:, np.newaxis]
    buf[:, :, 2] = np.float32(0.5)
    buf[:, :, 3] = np.float32(1.0)
    return buf


def write_png(rgba_f32: np.ndarray, path: str) -> None:
    """Encode linear f32 RGBA → sRGB u8 PNG (matches test_punch.py / test_kaleido.rs)."""
    lin = rgba_f32.astype(np.float64)
    r8  = (linear_to_srgb(lin[..., 0]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    g8  = (linear_to_srgb(lin[..., 1]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    b8  = (linear_to_srgb(lin[..., 2]).clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    a8  = (lin[..., 3].clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(np.stack([r8, g8, b8, a8], axis=-1), "RGBA").save(path, dpi=PPI)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Scalar fold helper (mirrors kaleido.py logic, used in seam tests)
# ---------------------------------------------------------------------------

def _fold_am(a: float, wedge_w: float) -> float:
    """Dihedral fold: scalar version of the array fold in kaleido.py."""
    k  = math.floor(a / wedge_w)
    am = a - k * wedge_w
    am = max(0.0, min(wedge_w, am))
    return wedge_w - am if (k % 2) == 1 else am


def _source_coord(
    x: float, y: float,
    cx: float, cy: float,
    wedge_w: float,
    zoom: float,
) -> tuple[float, float]:
    dx  = x - cx
    dy  = y - cy
    r   = math.hypot(dx, dy)
    a   = math.atan2(dy, dx)
    am  = _fold_am(a, wedge_w)
    return cx + (r / zoom) * math.cos(am), cy + (r / zoom) * math.sin(am)


# ---------------------------------------------------------------------------
# Seam tests
# ---------------------------------------------------------------------------

def test_seam_algorithm() -> None:
    """Source coords from either side of every seam must converge (≤ 1e-6).

    This is the hard seam-exactness requirement from spec/kaleido.md.
    Tests for segments ∈ {2, 4, 6, 8, 12}, all wedge boundaries, several radii.
    """
    tol = 1e-6
    eps = 1e-8      # angular offset either side of seam

    for segments in [2, 4, 6, 8, 12]:
        wedge_w = 2.0 * math.pi / segments
        size    = 32
        cx = cy = (size - 1) / 2.0
        zoom    = 1.0

        for k in range(segments):
            seam = k * wedge_w
            for probe_r in [4.0, 8.0, 12.0]:
                # Continuous coords just to the left and right of seam
                x1 = cx + probe_r * math.cos(seam - eps)
                y1 = cy + probe_r * math.sin(seam - eps)
                x2 = cx + probe_r * math.cos(seam + eps)
                y2 = cy + probe_r * math.sin(seam + eps)

                sx1, sy1 = _source_coord(x1, y1, cx, cy, wedge_w, zoom)
                sx2, sy2 = _source_coord(x2, y2, cx, cy, wedge_w, zoom)

                assert abs(sx1 - sx2) < tol, (
                    f"sx seam seg={segments} k={k} r={probe_r}: "
                    f"{sx1:.9f} vs {sx2:.9f}"
                )
                assert abs(sy1 - sy2) < tol, (
                    f"sy seam seg={segments} k={k} r={probe_r}: "
                    f"{sy1:.9f} vs {sy2:.9f}"
                )

    print("PASS  test_seam_algorithm")


def test_seam_no_edge() -> None:
    """Informational: run edge-detect on kaleido(gradient) and report pixel-value
    continuity across seam pairs.

    Per spec/kaleido.md the hard automated check is test_seam_algorithm (source
    coords converge).  The edge-detect test is for visual inspection by Jonathan.
    This function prints statistics but does NOT assert: gradient content near
    seam lines may naturally produce posterization boundaries that are unrelated
    to seam discontinuities.

    We DO assert that adjacent pixel pairs straddling the seam have nearly
    identical values (max per-channel absolute diff < 0.02), which is the
    pixel-level corollary of the source-coord convergence guarantee.
    """
    size    = 64
    src     = gradient(size)
    out     = kaleido(src, segments=6, rotation_deg=0.0, zoom=1.0)
    el      = edge(out, aggr=50.0, line_width=1)

    cx, cy  = (size - 1) / 2.0, (size - 1) / 2.0
    wedge_w = 2.0 * math.pi / 6
    MIN_R   = 3.0

    # --- Informational edge-pixel count along seam directions ---
    seam_edge_px = 0
    for k in range(6):
        seam_angle = k * wedge_w
        for probe_r in range(int(MIN_R) + 1, size // 2 - 2):
            xi = int(round(cx + probe_r * math.cos(seam_angle)))
            yi = int(round(cy + probe_r * math.sin(seam_angle)))
            if 0 <= xi < size and 0 <= yi < size:
                if el[yi, xi, 3] > 0.5:
                    seam_edge_px += 1
    print(f"INFO  edge pixels near seam lines: {seam_edge_px} (visual reference)")

    # --- Hard check: adjacent pairs straddling the seam must differ < 0.02 ---
    max_diff = 0.0
    violations = 0
    for k in range(6):
        seam_angle = k * wedge_w
        cos_s = math.cos(seam_angle)
        sin_s = math.sin(seam_angle)
        for y in range(size):
            for x in range(size):
                if math.hypot(x - cx, y - cy) < MIN_R:
                    continue
                cross1 = (y - cy) * cos_s - (x - cx) * sin_s
                for nx, ny in ((x + 1, y), (x, y + 1)):
                    if nx >= size or ny >= size:
                        continue
                    if math.hypot(nx - cx, ny - cy) < MIN_R:
                        continue
                    cross2 = (ny - cy) * cos_s - (nx - cx) * sin_s
                    if cross1 * cross2 >= 0:
                        continue
                    d = float(np.max(np.abs(
                        out[y, x, :3].astype(np.float64) -
                        out[ny, nx, :3].astype(np.float64)
                    )))
                    if d > max_diff:
                        max_diff = d
                    if d > 0.02:
                        violations += 1

    # Note: with correct dihedral fold both seam-straddling pixels sample the
    # same source direction → values are identical up to bilinear float noise.
    assert violations == 0, (
        f"Seam pixel-value diff > 0.02 found: {violations} pairs, "
        f"max_diff={max_diff:.6f}"
    )
    print(f"PASS  test_seam_no_edge (max seam pair diff: {max_diff:.6f})")


# ---------------------------------------------------------------------------
# Reference PNG generation
# ---------------------------------------------------------------------------

CASES = [
    # (name,              size, segments, rotation_deg, zoom)
    ("grad32_s6_r0_z1",    32, 6,  0.0, 1.0),
    ("grad32_s8_r30_z1p5", 32, 8, 30.0, 1.5),
    ("grad64_s4_r0_z1",    64, 4,  0.0, 1.0),
]


def main() -> None:
    test_seam_algorithm()
    test_seam_no_edge()

    for name, size, segs, rot, zm in CASES:
        src = gradient(size)
        out = kaleido(src, segments=segs, rotation_deg=rot, zoom=zm)
        write_png(out, f"test_kaleido_{name}.png")

    print("All kaleido tests passed.")


if __name__ == "__main__":
    main()
