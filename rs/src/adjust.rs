//! Pixel-level adjustment operations (invert, brightness, hue, …).
//!
//! Template: to add a new adjustment, write a public `fn name(rgba, mask, …)` that
//! delegates to `apply_adjustment` with the appropriate per-pixel RGB closure.
//! No changes to `apply_adjustment` itself are needed.
//!
//! Buffers: linear-light straight-alpha f32 (layer-native format, W×H×4).
//! Alpha is NEVER modified by any adjustment here.
//! Selection mask: `None` = whole layer; `Some(mask)` = blend old→new through mask.

use crate::layers::{linear_to_srgb_f64, srgb_to_linear};

// ---------------------------------------------------------------------------
// Generic wrapper
// ---------------------------------------------------------------------------

/// Apply a per-pixel RGB transform through the selection mask.
///
/// `transform`: receives linear `[R, G, B]` for one pixel, returns the adjusted triplet.
/// `mask`: `None` → treat every pixel as fully selected (whole-layer apply).
///         `Some` → per-pixel weight 0.0–1.0; lerp(old, new, mask[i]) per channel.
///
/// Alpha is never touched. Skips pixels where mask ≤ 0 for speed.
pub fn apply_adjustment(
    rgba:      &mut [f32],
    mask:      Option<&[f32]>,
    transform: impl Fn([f32; 3]) -> [f32; 3],
) {
    let n_px = rgba.len() / 4;
    for i in 0..n_px {
        let s = i * 4;
        let m = mask.map_or(1.0_f32, |mk| mk[i]);
        if m <= 0.0 { continue; }
        let [nr, ng, nb] = transform([rgba[s], rgba[s + 1], rgba[s + 2]]);
        rgba[s    ] = rgba[s    ] * (1.0 - m) + nr * m;
        rgba[s + 1] = rgba[s + 1] * (1.0 - m) + ng * m;
        rgba[s + 2] = rgba[s + 2] * (1.0 - m) + nb * m;
        // rgba[s + 3]: alpha untouched
    }
}

// ---------------------------------------------------------------------------
// Invert
// ---------------------------------------------------------------------------

/// Perceptual (sRGB) invert of one linear-light channel value.
/// Converts linear → sRGB, flips (1 − sRGB), converts back to linear.
/// Matches GIMP/Photoshop behavior; mid-grey maps to mid-grey.
#[inline]
fn invert_channel(v: f32) -> f32 {
    let s = linear_to_srgb_f64(v.clamp(0.0, 1.0) as f64).clamp(0.0, 1.0);
    srgb_to_linear((1.0 - s) as f32)
}

/// Perceptual (sRGB) invert applied through the selection mask.
///
/// `mask` = `None` → invert the whole layer.
/// `mask` = `Some` → invert only where mask > 0, soft-edge blend where 0 < mask < 1.
pub fn invert(rgba: &mut [f32], mask: Option<&[f32]>) {
    apply_adjustment(rgba, mask, |[r, g, b]| {
        [invert_channel(r), invert_channel(g), invert_channel(b)]
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ic(v: f32) -> f32 { super::invert_channel(v) }

    #[test]
    fn invert_channel_extremes() {
        let r0 = ic(0.0_f32);
        let r1 = ic(1.0_f32);
        println!("invert_channel(0.0) = {r0}  (want 1.0)");
        println!("invert_channel(1.0) = {r1}  (want 0.0)");
        assert!((r0 - 1.0).abs() < 1e-5, "ic(0) = {r0}, expected 1.0");
        assert!((r1 - 0.0).abs() < 1e-5, "ic(1) = {r1}, expected 0.0");
    }

    #[test]
    fn invert_green_cyan_red_no_mask() {
        // pure green linear (0,1,0) → magenta (1,0,1)
        let mut green = vec![0.0_f32, 1.0, 0.0, 1.0];
        invert(&mut green, None);
        println!("invert(green) → ({}, {}, {})", green[0], green[1], green[2]);

        // pure cyan linear (0,1,1) → red (1,0,0)
        let mut cyan = vec![0.0_f32, 1.0, 1.0, 1.0];
        invert(&mut cyan, None);
        println!("invert(cyan)  → ({}, {}, {})", cyan[0], cyan[1], cyan[2]);

        // pure red linear (1,0,0) → cyan (0,1,1)
        let mut red = vec![1.0_f32, 0.0, 0.0, 1.0];
        invert(&mut red, None);
        println!("invert(red)   → ({}, {}, {})", red[0], red[1], red[2]);

        assert!((green[0] - 1.0).abs() < 1e-4, "green R={}", green[0]);
        assert!((green[1] - 0.0).abs() < 1e-4, "green G={}", green[1]);
        assert!((green[2] - 1.0).abs() < 1e-4, "green B={}", green[2]);

        assert!((cyan[0] - 1.0).abs() < 1e-4, "cyan R={}", cyan[0]);
        assert!((cyan[1] - 0.0).abs() < 1e-4, "cyan G={}", cyan[1]);
        assert!((cyan[2] - 0.0).abs() < 1e-4, "cyan B={}", cyan[2]);

        assert!((red[0] - 0.0).abs() < 1e-4, "red R={}", red[0]);
        assert!((red[1] - 1.0).abs() < 1e-4, "red G={}", red[1]);
        assert!((red[2] - 1.0).abs() < 1e-4, "red B={}", red[2]);
    }

    #[test]
    fn invert_cyan_mask_full() {
        // Same as above but with an explicit all-1.0 mask — exercises the blend path.
        let mut cyan = vec![0.0_f32, 1.0, 1.0, 1.0];
        let mask = vec![1.0_f32];
        invert(&mut cyan, Some(&mask));
        println!("invert(cyan, mask=1.0) → ({}, {}, {})", cyan[0], cyan[1], cyan[2]);
        assert!((cyan[0] - 1.0).abs() < 1e-4, "R={}", cyan[0]);
        assert!((cyan[1] - 0.0).abs() < 1e-4, "G={}", cyan[1]);
        assert!((cyan[2] - 0.0).abs() < 1e-4, "B={}", cyan[2]);
    }
}
