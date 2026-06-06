//! Dihedral kaleidoscope transform — see spec/kaleido.md.
//! Mirrors py/kaleido.py exactly.

use crate::layers::{BlendMode, Layer};
use crate::sample::sample_bilinear;
use std::f64::consts::{PI, TAU};

/// Apply dihedral kaleidoscope to `source`, return a new layer.
///
/// * `segments`     — number of dihedral wedges (2–24)
/// * `rotation_deg` — pattern rotation in degrees (0–360)
/// * `zoom`         — >1 samples a smaller central source region (0.25–4.0)
pub fn kaleido(source: &Layer, segments: u32, rotation_deg: f64, zoom: f64) -> Layer {
    let w  = source.width  as usize;
    let h  = source.height as usize;
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;
    let rotation = rotation_deg * PI / 180.0;
    let wedge_w  = TAU / segments as f64;

    let mut out_rgba = vec![0.0f32; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;
            let r  = dx.hypot(dy);
            let a  = dy.atan2(dx) + rotation;

            // --- Dihedral fold ---
            let k  = (a / wedge_w).floor();
            let am = (a - k * wedge_w).clamp(0.0, wedge_w);
            let am_fold = if (k as i64).rem_euclid(2) == 1 {
                wedge_w - am
            } else {
                am
            };

            // --- Source coordinates ---
            let sx = cx + (r / zoom) * am_fold.cos();
            let sy = cy + (r / zoom) * am_fold.sin();

            // --- Bilinear sample (edge-clamped, f64 arithmetic) ---
            let samp = sample_bilinear(&source.rgba, w, h, sx, sy);
            let i = (y * w + x) * 4;
            out_rgba[i    ] = samp[0] as f32;
            out_rgba[i + 1] = samp[1] as f32;
            out_rgba[i + 2] = samp[2] as f32;
            out_rgba[i + 3] = samp[3] as f32;
        }
    }

    Layer {
        rgba:    out_rgba,
        width:   source.width,
        height:  source.height,
        mode:    BlendMode::Normal,
        opacity: 1.0,
        visible: true,
        name:    "kaleido".into(),
    }
}
