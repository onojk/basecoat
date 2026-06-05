# Technique: Band Generator (concentric outward fill)

## Purpose

Radiates alternating black/white bands outward from a seed layer's opaque
pixels, filling the canvas ring by ring. Each band lands on its own layer so
the original seed and all intermediate rings remain non-destructively editable.

## Inputs

| Input         | Description                                              |
|---------------|----------------------------------------------------------|
| Seed          | Active layer's opaque pixels (alpha > 0.5 = set pixel)   |
| Growth %      | Slider 0–100, default 30                                 |
| Fill %        | Slider 0–100, default 90                                 |

## Growth % → thickness mapping

```
thickness_px = floor(1.0 + (growth / 100.0) * 39.0 + 0.5)   // round-half-up
```

Range: growth=0 → 1 px, growth=30 → 13 px, growth=100 → 40 px.  
Rate is **constant** per iteration (no acceleration in v1).

Randomization envelope (Phase-2): growth ∈ [0, 100].

## Fill % — coverage metric and stop conditions

Coverage = (pixels in cumulative mask, including seed) / (total canvas pixels).

Stop when **either**:
1. `coverage ≥ fill_pct / 100.0` — threshold reached.
2. An iteration produces zero new pixels — canvas fully saturated (guard
   against infinite loop when bands can no longer expand).

Randomization envelope (Phase-2): fill ∈ [0, 100].

## Per-iteration algorithm

```
coverage_mask  ← opaque pixels of seed layer
coverage_count ← popcount(coverage_mask)
total_pixels   ← W × H
band_index     ← 0

loop:
    dilated  = dilate_chebyshev(coverage_mask, thickness_px)
    new_ring = dilated & ~coverage_mask

    if popcount(new_ring) == 0: break           // stop condition 2

    is_white       = (band_index % 2 == 0)      // white first
    append Band { mask: new_ring, white: is_white }

    coverage_mask  |= new_ring
    coverage_count += popcount(new_ring)
    band_index     += 1

    if coverage_count >= fill_pct / 100.0 * total_pixels: break  // stop 1
```

## Color cycle

Band 1 (first outward from seed) = **white**. Alternates B/W/B/W...  
Rationale: the seed is typically a black line layer; white first creates
immediate contrast against the black seed.

## Dilation primitive: Chebyshev (box) — two-pass 1D sliding window

Chebyshev dilation by `r` sets pixel (y,x) if any pixel in the rectangle
`[y−r..y+r, x−r..x+r]` was set in the input. Implemented as:
1. Horizontal pass: sliding window of width `2r+1` over each row.
2. Vertical pass: sliding window of height `2r+1` over each column.

Both passes run in O(W·H) regardless of `r`.

## Layer policy — protect existing work

The generator is the sole owner of the band layers it creates. It **never**
modifies layers that existed before Generate was pressed.

**Insertion order** (innermost first = topmost generator layer):
- Stack index order, high = top: `... | band_1 (white) | seed | upper_layers`
- Each new band inserts BELOW the current bottom-most generator band.
- After N bands: `... | band_N (outermost) | ... | band_1 | seed | ...`

**12-band cap** (own layers only):  
When inserting would bring the generator's layer count to 13, merge the 6
outermost (bottom-most) generator layers into one composite "merged" layer
before proceeding. The merged layer occupies 1 slot; count falls to 7 before
the new band is inserted (total 8). This can repeat if generation is very long.
Non-generator layers are never touched.

**One undo snapshot per Generate run.** A single structural checkpoint is taken
before any band is inserted. Pressing Undo once reverts the entire fill.

## Output layers

Each band layer: `W × H` RGBA f32 (linear light, straight alpha).
- Covered pixels: `(0,0,0,1)` black or `(1,1,1,1)` white.
- Uncovered pixels: `(0,0,0,0)` transparent.

## Validation

```bash
cd py  && python test_bands.py
cd rs  && cargo run --bin test_bands
# from repo root:
for name in dot_g30 dot_g10 dot_g50 diag_g30 confirm256; do
  python tools/pixdiff.py py/test_bands_${name}.png rs/test_bands_${name}.png
done
# Expected: all RESULT: PASS, max delta 0
```
