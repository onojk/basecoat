//! Shared bilinear sampler — mirrors py/sample.py exactly.
//!
//! Future transforms (rotate, scale, mirror) should call `sample_bilinear`
//! rather than re-implementing interpolation.

/// Bilinear-sample `src` at floating-point pixel coords `(sx, sy)`, edge-clamped.
///
/// * `src`  — row-major RGBA f32, `width × height × 4` elements.
/// * `w`, `h` — image dimensions.
/// * `sx`, `sy` — continuous pixel coords; clamped to `[0, w−1]` / `[0, h−1]`.
///
/// Returns `[f64; 4]` (RGBA).  All arithmetic is f64.
/// Out-of-bounds coords sample the border pixel (no transparent fringe).
pub fn sample_bilinear(src: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> [f64; 4] {
    let sx = sx.clamp(0.0, (w - 1) as f64);
    let sy = sy.clamp(0.0, (h - 1) as f64);

    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let tx = sx - x0 as f64;
    let ty = sy - y0 as f64;

    let p00 = px(src, w, x0, y0);
    let p10 = px(src, w, x1, y0);
    let p01 = px(src, w, x0, y1);
    let p11 = px(src, w, x1, y1);

    let mut out = [0.0f64; 4];
    for c in 0..4 {
        out[c] = (1.0 - tx) * (1.0 - ty) * p00[c] as f64
               +        tx  * (1.0 - ty) * p10[c] as f64
               + (1.0 - tx) *        ty  * p01[c] as f64
               +        tx  *        ty  * p11[c] as f64;
    }
    out
}

#[inline(always)]
fn px(src: &[f32], w: usize, x: usize, y: usize) -> &[f32] {
    let i = (y * w + x) * 4;
    &src[i..i + 4]
}
