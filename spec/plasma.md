# Technique: Plasma (Diamond-Square Midpoint Displacement)

## Output

| Property    | Value                                     |
|-------------|-------------------------------------------|
| Width       | 3500 px                                   |
| Height      | 3500 px                                   |
| Color model | RGBA 8 bpc, alpha = 255 (opaque)          |
| Resolution  | 600 PPI (pHYs = 23622 px/m, Unit::Meter)  |

## Algorithm

### Grid sizing

Diamond-square requires a `(2^n + 1) × (2^n + 1)` grid. For 3500 px output
use **n = 12 → 4097 × 4097**. After generation, crop the top-left 3500 × 3500
region. The remaining 597 columns/rows are discarded.

### Per-channel generation

Each of R, G, B is generated independently from its own seed (see below).
The grid is a 2D array of `f64` values in [0, 1).

1. Seed the four corners: `grid[0][0]`, `grid[0][N-1]`, `grid[N-1][0]`,
   `grid[N-1][N-1]` each get one `rng_unit()` draw.
2. Set `scale = turbulence`.
3. For `step` from `(N-1)` down to `1`, halving each iteration (`step /= 2`):
   - **Diamond step**: for every complete square of side `step`, set the center
     to `mean(4 corners) + displacement`.
   - **Square step**: for every diamond midpoint on the grid edges and
     diagonals, set it to `mean(available neighbours) + displacement`.  
     Border midpoints have only 3 neighbours — clamp, do not wrap.
   - `displacement = (rng_unit() * 2.0 - 1.0) * scale`
   - After both steps, `scale *= 0.5`.
4. Clamp all values to [0, 1].

### Draw-order contract (PRNG consumption)

Both implementations MUST consume random numbers in identical order. Deviation
causes diff failure.

- **Corner seeding**: row-major — `(0,0)`, `(0,N-1)`, `(N-1,0)`, `(N-1,N-1)`.
- **Diamond step**: row-major over center points (ascending row, then col
  within row).
- **Square step**: row-major over midpoints (ascending row, then col within
  row). Within each row, process only the midpoints that belong to that row.
- Each midpoint consumes exactly **1** random draw.
- Complete the full diamond pass, then the full square pass, before proceeding
  to the next level.

### Border clamping (no wrap)

Edge and corner midpoints in the square step may have fewer than 4 neighbours.
Sum only the available (in-bounds) neighbours and divide by their count.
**Do not wrap around.** This produces a mild darkening/brightening artifact
at borders, which is intentional and reproducible.

## PRNG: xorshift64\*

State is one `u64`. Advance function:

```
state ^= state >> 12
state ^= state << 25
state ^= state >> 27
result = state * 0x2545F4914F6CDD1D
```

`rng_unit()` returns the top 53 bits of `result` as `f64` divided by `2^53`:

```
(result >> 11) as f64 / 9007199254740992.0
```

### Per-channel seed derivation

From one user-supplied `seed: u64`:

```
seed_R = seed
seed_G = seed ^ 0x9E3779B97F4A7C15
seed_B = seed ^ 0xD1B54A32D192ED03
```

The constants are chosen to be maximally distant in u64 space (golden-ratio
and SplitMix64-derived). They ensure the three channels are decorrelated even
when `seed = 0` (seed_G and seed_B are never zero for any seed value, since
the XOR constants are non-zero).

## Color space

Channel values produced by diamond-square are **linear-light** (no gamma).
Convert to sRGB on encode:

```
if v <= 0.0031308 { v * 12.92 }
else              { 1.055 * v^(1/2.4) - 0.055 }
```

This intentionally differs from GIMP 2.10's plasma filter, which operates in
perceptual (sRGB) space throughout. Our plasma is physically correct for
linear compositing workflows.

Alpha channel is always 255 (fully opaque).

## Parameters

| Parameter   | Default | Description                                     |
|-------------|---------|------------------------------------------------|
| seed        | 0       | u64 base seed for all three channels            |
| turbulence  | 1.0     | Initial scale for random displacement           |

## Validation

```bash
cd py && python plasma.py               # → py/plasma.png
cd rs && cargo run --bin plasma         # → rs/plasma.png
python tools/pixdiff.py py/plasma.png rs/plasma.png
# Expected: max delta 0, diff pixels 0, RESULT: PASS
```
