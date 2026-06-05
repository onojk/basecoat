# Technique: Transparent Canvas

## Purpose

Baseline technique. Produces a fully-transparent RGBA canvas with correct
resolution metadata. All other techniques start from this foundation.

## Specification

| Property       | Value                          |
|----------------|-------------------------------|
| Width          | 3500 px                        |
| Height         | 3500 px                        |
| Color model    | RGBA (4 channels, 8 bpc)       |
| Fill           | (0, 0, 0, 0) — fully transparent |
| Resolution     | 600 PPI                        |
| pHYs chunk     | xppu = yppu = 23622, unit = meter |

### PPI → px/m conversion

```
px/m = round(PPI / 0.0254) = round(600 / 0.0254) = 23622
```

The PNG `pHYs` chunk stores integer pixels-per-unit. Using `Unit::Meter`
with 23622 encodes exactly 600 PPI (error < 0.01%).

## Color conventions

These conventions apply to every technique in this project:

**Gamma / transfer function**  
All pixel values are stored as **sRGB** (the PNG default). No linear-light
compositing is performed unless a technique spec explicitly says so. When a
technique requires linear compositing, convert to linear before blending and
back to sRGB before writing.

**Alpha interpretation**  
All files store **straight (unassociated) alpha**. Pillow's `Image.new` and
the `png` crate both write straight alpha by default. Premultiplied
intermediate buffers must be un-premultiplied before saving.

## Validation

Run both implementations and diff with tolerance 0:

```
cd py  && python canvas.py          # writes py/canvas.png
cd rs  && cargo run --bin canvas    # writes rs/canvas.png
python tools/pixdiff.py py/canvas.png rs/canvas.png
```

Expected output: `RESULT: PASS`, max delta 0, 0 differing pixels.
