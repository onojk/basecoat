# Technique: QBist (GIMP-faithful genetic abstract-pattern fill)

## Output

| Property    | Value                                     |
|-------------|-------------------------------------------|
| Width       | 3500 px (default; configurable)           |
| Height      | 3500 px (default; configurable)           |
| Color model | RGBA 8 bpc, alpha = 255 (opaque)          |
| Resolution  | 600 PPI (pHYs = 23622 px/m, Unit::Meter)  |
| Color space | **No** linear→sRGB conversion — qbist writes reg[0] bytes directly |

## Algorithm Overview

QBist generates abstract patterns by:
1. Creating a random "genome" (sequence of transform operations) from a seed.
2. Optimising the genome to remove unused transforms/registers.
3. For each pixel (with optional supersampling), initialising registers from
   pixel coordinates, applying the transform sequence, and reading the output
   from register 0.

## PRNG

### xorshift64\*

State is one `u64`. Advance function (same as plasma):

```
state ^= state >> 12
state ^= state << 25
state ^= state >> 27
result = state * 0x2545F4914F6CDD1D
rng_unit() = (result >> 11) as f64 / 9007199254740992.0   // top-53 / 2^53
```

### Seed avalanche (REQUIRED)

Before any xorshift draw, the user seed is avalanched through splitmix64:

```
z = seed + 0x9E3779B97F4A7C15
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z ^ (z >> 27)) * 0x94D049BB133111EB
initial_xorshift_state = z ^ (z >> 31)
```

**Without this step, seed=0 produces all-zero xorshift draws → degenerate genome.**

### rng_range(n)

```
rng_range(n) = floor(rng_unit() * n)
```

No rejection loop. For n ≤ 9 this gives uniform distribution over [0, n-1].

## Genome

Constants:
- `TRANSFORMS = 36`
- `REGISTERS  = 6`
- `OPCODES    = 9`

Arrays, all length 36:
- `transformSequence[36]` — opcode index ∈ [0, 8]
- `source[36]`            — source register ∈ [0, 5]
- `control[36]`           — control register ∈ [0, 5]
- `dest[36]`              — destination register ∈ [0, 5]

### create_info(seed)

Draw order: for k in 0..35 (inclusive), draw **ts, src, ctl, dst** in that
order (4 draws per k, 144 draws total):

```
rng = RNG(seed)           // splitmix64 avalanche applied in constructor
for k in 0..36:
    transformSequence[k] = rng_range(9)
    source[k]            = rng_range(6)
    control[k]           = rng_range(6)
    dest[k]              = rng_range(6)
```

## optimize()

### Step 1 — fixup degenerate transforms

```
for i in 0..36:
    if transformSequence[i] in {ROTATE, ROTATE2, COMPLEMENT}:
        control[i] = dest[i]
```

These opcodes ignore the control register; redirecting control→dest avoids
marking an extra register as "needed for initialisation".

### Step 2 — backward dependency walk

```
check_last_modified(index=36, reg=0)
```

```python
def check_last_modified(index, reg):
    # find last transform before `index` that writes to `reg`
    i = index - 1
    while i >= 0 and dest[i] != reg:
        i -= 1
    if i < 0:
        used_reg_flag[reg] = True   # reg must be initialised
    else:
        used_trans_flag[i] = True
        check_last_modified(i, source[i])
        check_last_modified(i, control[i])
```

Returns `(used_trans_flag[36], used_reg_flag[6])`.

## Transform Opcodes

For each transform, `sv = reg[source]`, `cv = reg[control]`, result → `reg[dest]`.
All arithmetic in **f64**.

| # | Name        | Formula (per channel c)                                               |
|---|-------------|-----------------------------------------------------------------------|
| 0 | PROJECTION  | scalar = (sv[0]·cv[0]+sv[1]·cv[1]+sv[2]·cv[2])/3 ; dst[c]=scalar·cv[c] |
| 1 | SHIFT       | v=sv[c]+cv[c] ; dst[c] = v>1.0 ? v-1.0 : v                          |
| 2 | SHIFTBACK   | v=cv[c]-sv[c] ; dst[c] = v<0.0 ? v+1.0 : v                          |
| 3 | ROTATE      | dst = (sv[1], sv[2], sv[0])                                           |
| 4 | ROTATE2     | dst = (sv[2], sv[0], sv[1])                                           |
| 5 | MULTIPLY    | dst[c] = sv[c]·cv[c]                                                  |
| 6 | COMPLEMENT  | dst[c] = 1.0−sv[c]                                                    |
| 7 | SINE        | dst[c] = (1.0+sin(20.0·sv[c]))/2.0                                   |
| 8 | CONDITIONAL | if cv[0]+cv[1]+cv[2] > 0.5 : dst=sv else dst=cv                      |

Notes:
- ROTATE/ROTATE2/COMPLEMENT: control register is unused (fixup sets it to dest).
- SINE: uses only sv; cv is unused.
- Values are **not clamped** between transforms.

## Rendering

```
for each pixel (col, row) in 0..W × 0..H:
    accum[3] = {0, 0, 0}

    for ys in 0..os:
        for xs in 0..os:

            // Initialise flagged registers
            x_norm = (col * os + xs) / (W * os)        // f64
            y_norm = (row * os + ys) / (H * os)        // f64
            for i, r in enumerate(used_reg_list):      // i = 0, 1, 2, …
                reg[r] = (x_norm, y_norm, i / 6.0)

            // Apply flagged transforms in order
            for t in used_trans_list:
                apply opcode[t] with source[t], control[t], dest[t]

            // Inner quantisation (GIMP truncation)
            for c in 0..3:
                accum[c] += trunc(reg[0][c] * 255.0 + 0.5)

    // Outer quantisation
    for c in 0..3:
        out[c] = clamp(floor(accum[c] / (os*os) + 0.5), 0, 255) as u8
    out[3] = 255  // alpha
```

**trunc** means truncation toward zero (C cast semantics, not floor).  
Do the `+0.5` and `floor`/`trunc` explicitly in f64 in both implementations.

## Parameters

| Parameter    | Default | Description                            |
|--------------|---------|----------------------------------------|
| seed         | 0       | u64 seed (accepts 0x hex)              |
| oversampling | 4       | Supersampling factor (os × os samples) |
| size         | 3500    | Output width and height in pixels      |

## Validation

```bash
cd py && python qbist.py && python test_qbist.py
cd rs && cargo run --release --bin qbist && cargo run --release --bin test_qbist
for n in seed0_os1 seed0_os4 seed1_os4 seed42_os4 confirm3500; do
    python tools/pixdiff.py py/test_qbist_${n}.png rs/test_qbist_${n}.png
done
python tools/pixdiff.py py/qbist.png rs/qbist.png
# All must report: max delta 0, RESULT: PASS
```

## Reference Test Cases

| Name                  | Size | Seed | OS |
|-----------------------|------|------|----|
| test_qbist_seed0_os1  | 256  | 0    | 1  |
| test_qbist_seed0_os4  | 256  | 0    | 4  |
| test_qbist_seed1_os4  | 256  | 1    | 4  |
| test_qbist_seed42_os4 | 256  | 42   | 4  |
| test_qbist_confirm3500| 3500 | 0    | 4  |
