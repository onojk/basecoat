# Technique: Punch (Contrast + Saturation grouped pass)

## Purpose

A destructive grouped adjustment applied to the active layer N times.
One "punch" = contrast S-curve followed by saturation boost, compounded
`passes` times. Jonathan's habitual move: contrast=9.0, saturation=4.0, passes=6.

## Operation

Punch is **not** a generator and **not** a blend mode. It modifies the RGB
channels of an existing layer buffer in place. Alpha is left untouched.

The three controls are applied as a single grouped operation; one APPLY button
runs all passes and pushes **one undo snapshot**.

## Per-pass math (operate in linear light)

Process only RGB; leave alpha unchanged. Within each pass:

### Step 1 — Contrast (sigmoid S-curve, per channel)

```
s(x)  = 1.0 / (1.0 + exp(−k · (x − 0.5))

lo    = s(0.0)
hi    = s(1.0)
out   = (s(in) − lo) / (hi − lo)
```

`k` is the contrast strength. `k = 9.0` is the maximum / "100%" value.

**Identity guard**: if `k < 1e-6`, skip the sigmoid and pass through unchanged
(the formula degenerates as k→0; guard avoids the divide-by-near-zero in
`hi − lo`).

The normalization `(s(x) − lo) / (hi − lo)` ensures exact 0→0 and 1→1
mapping so the output stays in [0, 1] for input in [0, 1].

### Step 2 — Saturation (pull away from luma, per channel)

```
luma   = 0.2126·r + 0.7152·g + 0.0722·b   // BT.709, on post-contrast RGB
out_c  = luma + sat · (c − luma)           // per channel c ∈ {r, g, b}
```

`sat = 1.0` is identity, `sat = 0.0` converts to greyscale, `sat > 1.0` boosts.
`sat = 4.0` is the maximum / "100%" value.

**Clamp** all three channels to [0, 1] after saturation. This ensures the next
pass receives valid input.

Luma is recomputed from the post-contrast RGB on each pass.

## Pass compounding

```
for i in 0..passes:
    apply_contrast_to_rgb(buf, k)
    apply_saturation_to_rgb(buf, sat)   // uses post-contrast rgb for luma
    clamp_rgb(buf)
```

`passes` default = 6. Each iteration reads the clamped output of the previous
pass.

## Parameter ranges (randomization envelopes for Phase-2)

| Parameter     | Min  | Max  | Default | Identity value |
|---------------|------|------|---------|---------------|
| contrast (k)  | 0.0  | 9.0  | 9.0     | 0.0           |
| saturation    | 0.0  | 4.0  | 4.0     | 1.0           |
| passes        | 1    | 6    | 6       | —             |

These ranges are declared envelopes for automated randomization (Phase-2).

## Floating-point model

Computation in **f64** in both Python and Rust.  
Read path: layer buffer (f32) → promote to f64 at pass start.  
Write path: clamp f64 result → demote to f32 at pass end (IEEE 754
round-to-nearest-even).  
Subsequent passes operate on the f32-quantised values.

This mirrors the `layers.rs` composite model and gives identical results
across scalar (Rust `f64::exp`) and numpy (libm `exp`) on Linux.

## Color space

Buffers store **linear-light** RGBA (straight alpha), same as all other
basecoat operations. No gamma conversion is performed inside punch.

## Validation

```bash
cd py  && python test_punch.py      # writes py/test_punch_*.png
cd rs  && cargo run --bin test_punch # writes rs/test_punch_*.png
# from repo root:
for name in grad_default grad_identity grad_grayscale grad_1pass plasma_default 3500; do
  python tools/pixdiff.py py/test_punch_${name}.png rs/test_punch_${name}.png
done
# Expected: all RESULT: PASS, max delta 0
```
