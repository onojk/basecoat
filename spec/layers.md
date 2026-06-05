# Layer Stack + Compositor Engine

## Layer model

A **Layer** contains:

| Field   | Type                        | Notes                                         |
|---------|-----------------------------|-----------------------------------------------|
| rgba    | 3500×3500 RGBA buffer       | Straight (unassociated) alpha, linear-light   |
| mode    | BlendMode                   | See blend modes below                         |
| opacity | f32 in [0, 1]               | Multiplied into the layer's alpha channel     |
| visible | bool                        | If false, layer is skipped in composite       |
| name    | String                      | Human-readable label                          |

A **Stack** is an ordered list of Layers, index 0 = bottom. Hard cap
**MAX_LAYERS = 25**. Attempts to add a 26th layer are rejected (error return /
no-op); existing layers are unchanged.

Layers are **destructive**: operations bake directly into pixels. No
adjustment layers, no non-destructive FX, no effect chains.

## Blend modes (this release)

Five modes ship in this slice: **Normal, Multiply, Screen, Overlay, Difference**.

All blend math operates in **linear-light** floating-point. This matches
GIMP's non-legacy "default" modes. GIMP also ships "(legacy)" variants that
operate in perceptual (sRGB) space — this project does NOT implement those.

Let `a` = top layer linear value, `b` = bottom layer linear value, per channel:

| Mode       | Formula (per channel, linear)                       |
|------------|-----------------------------------------------------|
| Normal     | `a`                                                 |
| Multiply   | `a * b`                                             |
| Screen     | `1 - (1 - a) * (1 - b)`                             |
| Overlay    | `b < 0.5 ? 2·a·b : 1 - 2·(1-a)·(1-b)`              |
| Difference | `|a - b|`                                           |

Overlay pivots on **b** (the bottom layer's value), not a.

## Compositing formula (straight-alpha source-over)

The blend mode formula computes **blended color**. Coverage is handled
separately by standard Porter-Duff source-over using the top layer's
**effective alpha** `As = alpha_channel * opacity`:

```
Aout = As + Ab * (1 - As)
Cout = (blend(Cs, Cb) * As  +  Cb * Ab * (1 - As)) / Aout
       where Cout = 0 when Aout = 0
```

`Cs` = top layer linear color (pre-divided by alpha — buffer stores straight),
`Cb` = bottom composite linear color,
`Ab` = bottom composite alpha.

The blend mode affects **color**; the alpha-over formula handles **coverage**.
These are two distinct operations applied in sequence.

## composite(stack) → RGBA image

Start from a fully-transparent black pixel `(0, 0, 0, 0)`.  
Iterate layers bottom → top (index 0 first). Skip layers where `visible = false`.  
For each visible layer apply the over-composite formula above.  
Return the accumulated RGBA image.

## flatten_visible(stack)

Runs `composite(stack)` over the visible layers, then:

1. Removes all **visible** layers from the stack.
2. Inserts the composite result as a single new layer at the position of the
   former bottom-most visible layer, with mode=Normal, opacity=1, visible=true.
3. Invisible layers remain in their original positions relative to the new
   merged layer (those below it stay below, those above it stay above).

## Undo ring (depth 20)

Undo history is a **bounded ring buffer** of depth 20. Oldest entries are
silently dropped when the ring is full.

Snapshot strategy — per operation type:

| Operation                           | What is snapshotted                          |
|-------------------------------------|----------------------------------------------|
| Pixel op (fill, blend on one layer) | That layer's pixel buffer only               |
| Structural op (add/delete/reorder/flatten) | Full layer list + any destroyed buffers |

`undo()` pops the most recent snapshot and restores it. Undo past the oldest
snapshot is a no-op. This is an engine-level history; a GUI wires to it later.

## Color space conventions

- Pixel buffers store **linear-light** RGBA (straight alpha).
- sRGB ↔ linear conversion uses the standard piecewise IEC 61966-2-1 formula
  (same as `spec/plasma.md`).
- On PNG encode: convert linear → sRGB per channel, alpha passes through unchanged.
- On PNG decode: convert sRGB → linear per channel, alpha passes through unchanged.
