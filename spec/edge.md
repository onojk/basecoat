# Technique: Edge (posterize-then-outline)

## Purpose

Finds boundaries between distinct color regions in the active layer and draws
fixed-width black lines on those boundaries, on a NEW transparent layer placed
above the source. The resulting layer is the canonical seed for the Band
Generator and future ops.

## Controls

| Parameter       | Range   | Default | Meaning                                    |
|-----------------|---------|---------|--------------------------------------------|
| Aggressiveness  | 0–100 % | 50 %    | Maps to posterize level count (see below)  |
| Line width      | 1–7 px  | 3 px    | Width of drawn line (see dilation)         |

Randomization envelopes (Phase-2): aggressiveness ∈ [0, 100], line_width ∈ [1, 7].

## Algorithm (linear-light, operates on f32 layer buffer)

### Step 1 — Posterize

Map each RGB channel of each pixel to a bucket index 0 … N-1:

```
bucket(c_f32, N) = clamp( floor(c_f32_as_f64 × (N−1) + 0.5), 0, N−1 )
```

The `+0.5` before `floor` implements round-half-up, matching Rust's `f32::round`
semantics and avoiding numpy banker's-rounding at exact half-integer values.

**Aggressiveness → N mapping** (also round-half-up):

```
N = floor( 3.0 + (aggr / 100.0) × 29.0 + 0.5 )
```

Range: aggr=0 → N=3 (coarse, only boldest boundaries),
       aggr=100 → N=32 (fine, dense lines).

### Step 2 — Boundary detection (4-neighbor)

A pixel is an edge pixel if **any** of its four axis-aligned neighbours
(up/down/left/right) has a different (br, bg, bb) bucket triple.

**Border handling**: out-of-image neighbours are treated as identical to the
pixel itself — no edge is drawn at the image frame.

### Step 3 — Dilation

Dilate the 1-px edge mask using the **same** `dilate_chebyshev` primitive from
`rs/src/bands.rs` (Python: `bands.dilate_chebyshev`).

```
radius = floor(line_width / 2)   // integer division
```

| width | radius | actual rendered width |
|-------|--------|-----------------------|
| 1     | 0      | 1 px (no dilation)    |
| 3     | 1      | 3 px                  |
| 5     | 2      | 5 px                  |
| 7     | 3      | 7 px                  |

Even widths (2, 4, 6) give the next-odd width (3, 5, 7) because dilation by r
expands a 1-px feature to (2r+1) px. This is intentional and documented.

### Step 4 — Output layer

New layer ABOVE the source, same dimensions (W × H), RGBA:
- Edge pixels: `(0, 0, 0, 1)` linear — opaque black.
- All other pixels: `(0, 0, 0, 0)` — fully transparent.

The source layer is **never modified**.

## Layer policy

- Edge always creates a fresh layer above the active one.
- Pre-existing layers are never modified, flattened, or moved.
- One undo snapshot (structural checkpoint) before insertion.

## Validation

```bash
cd py  && python test_edge.py
cd rs  && cargo run --bin test_edge
# from repo root:
for name in split_a50_w1 split_a50_w3 split_a50_w7 \
            quad_a50_w3 quad_a0_w3 quad_a100_w3 \
            plasma_a50_w3 confirm3500; do
  python tools/pixdiff.py py/test_edge_${name}.png rs/test_edge_${name}.png
done
# Expected: all RESULT: PASS, max delta 0
```
