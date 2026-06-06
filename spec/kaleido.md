# Technique: Kaleidoscope (dihedral)

## Purpose

Composites the marked layer set (or the active layer alone if none are marked)
into a single source RGBA image, then maps it through a dihedral (mirrored)
kaleidoscope transform into a **NEW** layer above the topmost marked (or active)
layer.  Adjacent wedges are exact mirror images, so wedge seams have zero
discontinuity.

## Controls

| Parameter | Range     | Default | Meaning                                        |
|-----------|-----------|---------|------------------------------------------------|
| Segments  | 2–24      | 6       | Number of dihedral wedges                      |
| Rotation  | 0–360 °   | 0       | Rotates the pattern; shifts sampling direction |
| Zoom      | 0.25–4.0  | 1.0     | >1 samples a smaller central source region     |

Center is fixed at canvas center for v1 (click-to-place deferred).

Randomization envelopes (Phase-2): segments ∈ [2, 24], rotation ∈ [0, 360],
zoom ∈ [0.25, 4.0].

## Source compositing

- Collect marked layers in ascending stack-index order (bottom-first).
- If ≥ 1 marked: composite them via the existing compositor into one RGBA image.
- If 0 marked: use the active layer alone; note this in the status bar.
- Originals are never modified.

## Algorithm (dihedral kaleidoscope)

Canvas center: `cx = (W−1)/2`, `cy = (H−1)/2` (float64).  
`rotation_radians = rotation_degrees × π / 180`.

For each output pixel (x, y):

1. `dx = x−cx`, `dy = y−cy`; `r = hypot(dx, dy)`; `a = atan2(dy, dx) + rotation_radians`
2. Wedge width: `w = 2π / segments`
3. Dihedral fold:
   ```
   k  = floor(a / w)          # wedge index; may be negative
   am = a − k × w             # positive-modulo residual ∈ [0, w] (fp)
   am = clamp(am, 0, w)       # guard against fp slop at exact multiples
   if k rem_euclid 2 == 1:    # odd wedge → mirror
       am = w − am
   ```
4. Source coordinates (`zoom > 1` samples a smaller central region):
   ```
   sx = cx + (r / zoom) × cos(am)
   sy = cy + (r / zoom) × sin(am)
   ```
5. Bilinear-sample the source image at `(sx, sy)` with **edge-clamping**
   (see § Shared Sampler).
6. Write sampled RGBA (f64 → f32) to output pixel.

All transcendental math (`atan2`, `hypot`, `cos`, `sin`) and all accumulation
are done in **f64** in both Python and Rust.

## Seam-exactness requirement (hard)

At every wedge boundary (`a = k × w`) the fold maps both adjacent wedges to
the same source direction:

- Left-of-boundary (wedge k−1 is odd, am_raw → w):
  `am = w − w = 0` → source direction 0 ✓
- Right-of-boundary (wedge k is even, am → 0):
  `am = 0` → source direction 0 ✓  
  (and the complementary even/odd case gives both sides → `w`, same direction ✓)

The fold is therefore C0-continuous at every boundary; there is **zero pixel
discontinuity** at wedge seams.

### Automated seam check

For each seam angle `θ = k × wedge_w`, at several radii, compute source coords
from angle `θ − ε` and `θ + ε` (ε = 1 × 10⁻⁸ rad). Both must agree within
1 × 10⁻⁶ in x and y.

Implemented in `py/test_kaleido.py :: test_seam_algorithm` and
`rs/src/bin/test_kaleido.rs :: test_seam_algorithm`.

### Edge-detect acceptance test

Apply the Edge technique (aggr = 50, width = 1) to the kaleidoscope output of a
smooth gradient source. **No edge pixel must appear along any wedge seam line.**
Implemented in `py/test_kaleido.py :: test_seam_no_edge`.
(Visual verification by Jonathan; automated pixel-level check above.)

Seam correctness is a **hard** requirement — it is **not** subject to the f64
diff tolerance posture below.

## Shared Sampler

Modules: `rs/src/sample.rs` / `py/sample.py`

```
# Rust
pub fn sample_bilinear(src: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> [f64; 4]

# Python  
def sample_bilinear(src: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray
    # src (H,W,C) float32; sx,sy (H,W) float64; returns (H,W,C) float64
```

Algorithm (all f64):

1. Clamp `sx` to `[0, w−1]`, `sy` to `[0, h−1]`.
2. `x0 = floor(sx)`, `x1 = min(x0+1, w−1)`; `y0 = floor(sy)`, `y1 = min(y0+1, h−1)`.
3. `tx = sx − x0`, `ty = sy − y0`.
4. `result = (1−tx)(1−ty)·src[y0,x0] + tx(1−ty)·src[y0,x1] + (1−tx)ty·src[y1,x0] + tx·ty·src[y1,x1]`

Edge behaviour: clamped — samples the border pixel rather than introducing a
transparent fringe.

## Layer policy

- One structural undo checkpoint (`stack.checkpoint()`) before insertion.
- New layer inserted **above** the topmost marked (or active) layer, becomes active.
- Marked set is cleared after apply.
- Thumbnail vecs kept in sync via `thumb_insert`.

## f64 diff tolerance posture

`atan2`, `hypot`, `cos`, `sin`, and bilinear weights all use transcendental or
fractional arithmetic. NumPy and Rust libm may differ in the last bit on these
operations, and every interior pixel passes through interpolation. True delta-0
across the two languages may be unachievable.

**Posture**: compute in f64 in both; attempt delta-0 on small grids; if it fails
only by tiny per-channel amounts (≤ 2 LSB = ≤ 2 in u8 space after
round-and-clamp) spread uniformly across the image (not structured), that is
interpolation/libm noise and is **acceptable**. Use `--tolerance 2` in pixdiff.
Document the observed max delta.

This tolerance applies **only** to the py-vs-rs comparison and is entirely
separate from the seam requirement.

## Validation

```bash
cd py && python test_kaleido.py          # seam tests + reference PNGs
cd rs && cargo run --bin test_kaleido    # seam test + reference PNGs
# from repo root:
for name in grad32_s6_r0_z1 grad32_s8_r30_z1p5 grad64_s4_r0_z1; do
  python tools/pixdiff.py py/test_kaleido_${name}.png rs/test_kaleido_${name}.png --tolerance 2
done
```
