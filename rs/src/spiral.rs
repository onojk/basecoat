//! Two-tone spiral generator — Logarithmic (nautilus) or Archimedean (even-spaced).
//!
//! **Logarithmic** (default): r = exp(b·θ_total), windings widen proportionally
//! from centre — the nautilus / golden-spiral look.  `turns` full windings span
//! centre→canvas-corner; b is derived so the geometry matches exactly.
//!
//! **Archimedean**: constant winding pitch (even spacing every `spacing` px);
//! windings same width at all radii — the GIMP default look.
//!
//! Phase formula — logarithmic:
//!   EPS          = 1.0  (r floor; guards ln(0) at exact centre)
//!   b            = ln(max_r / EPS) / (2π · turns)
//!   spiral_angle = ln(max(r, EPS) / EPS) / b
//!   phase        = (spiral_angle − θ) / 2π · arms
//!   frac         = phase mod 1.0 ∈ [0, 1)
//!   pixel = A if frac < 0.5, else B
//!
//! Phase formula — Archimedean:
//!   spacing = max_r / turns
//!   phase   = r / spacing − (θ / 2π) · arms
//!   frac    = phase mod 1.0
//!   pixel = A if frac < 0.5, else B
//!
//! NxN supersampling (`OVERSAMPLING`) averages linear colours across sub-samples
//! — boundary is smooth (no staircase) while interiors stay pure A or B.

use std::f64::consts::PI;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

pub const OVERSAMPLING: usize = 3;

const TAU: f64 = 2.0 * PI;
const EPS: f64 = 1.0; // r floor for logarithmic singularity guard

/// Spiral geometry selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpiralKind {
    Logarithmic,
    Archimedean,
}

/// Render a two-tone spiral into `buf` (W×H×4 linear RGBA, alpha = 1 throughout).
///
/// - `color_a` / `color_b`: linear-light \[R,G,B,A\].
/// - `kind`: Logarithmic (nautilus) or Archimedean (even-spaced).
/// - `turns`: windings from centre to canvas corner (controls density for both kinds).
/// - `arms`: interleaved spiral starts (1 = single continuous coil).
/// - `oversampling`: NxN sub-pixel samples per output pixel.
/// - `progress`: incremented once per completed row for the UI progress bar.
#[allow(clippy::too_many_arguments)]
pub fn render_with_progress(
    buf:          &mut [f32],
    w:            usize,
    h:            usize,
    color_a:      [f32; 4],
    color_b:      [f32; 4],
    kind:         SpiralKind,
    turns:        u32,
    arms:         u32,
    oversampling: usize,
    progress:     &Arc<AtomicU32>,
) {
    let cx      = w as f64 / 2.0;
    let cy      = h as f64 / 2.0;
    let arms_f  = arms as f64;
    let os      = oversampling as f64;
    let os2_inv = 1.0 / (os * os);
    let max_r   = cx.hypot(cy); // centre-to-corner
    let turns_f = turns.max(1) as f64;

    // Pre-compute kind-specific constants once outside the pixel loop.
    let log_b  = (max_r / EPS).ln() / (TAU * turns_f); // logarithmic growth rate
    let sp_inv = turns_f / max_r.max(EPS);              // 1/spacing for Archimedean

    let ca = [color_a[0] as f64, color_a[1] as f64, color_a[2] as f64];
    let cb = [color_b[0] as f64, color_b[1] as f64, color_b[2] as f64];

    for row in 0..h {
        for col in 0..w {
            let mut sr = 0.0_f64;
            let mut sg = 0.0_f64;
            let mut sb = 0.0_f64;

            for sy in 0..oversampling {
                for sx in 0..oversampling {
                    let px = col as f64 + (sx as f64 + 0.5) / os;
                    let py = row as f64 + (sy as f64 + 0.5) / os;
                    let dx = px - cx;
                    let dy = py - cy;
                    let r     = (dx * dx + dy * dy).sqrt();
                    let theta = dy.atan2(dx);

                    let frac = match kind {
                        SpiralKind::Logarithmic => {
                            let rr           = r.max(EPS);
                            let spiral_angle = (rr / EPS).ln() / log_b;
                            let phase        = (spiral_angle - theta) / TAU * arms_f;
                            phase - phase.floor()
                        }
                        SpiralKind::Archimedean => {
                            let phase = r * sp_inv - (theta / TAU) * arms_f;
                            phase - phase.floor()
                        }
                    };

                    let [cr, cg, cb_] = if frac < 0.5 { ca } else { cb };
                    sr += cr;
                    sg += cg;
                    sb += cb_;
                }
            }

            let i = (row * w + col) * 4;
            buf[i    ] = (sr * os2_inv) as f32;
            buf[i + 1] = (sg * os2_inv) as f32;
            buf[i + 2] = (sb * os2_inv) as f32;
            buf[i + 3] = 1.0;
        }
        progress.fetch_add(1, Ordering::Relaxed);
    }
}
