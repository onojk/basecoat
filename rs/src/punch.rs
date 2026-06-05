//! Punch adjustment — contrast S-curve + saturation, applied N passes.
//! See spec/punch.md.  Float model: compute in f64, store as f32.

/// Apply punch in-place to a flat RGBA f32 buffer (row-major, 4 floats/pixel).
/// Alpha is untouched.  `k` = contrast strength [0..9], `sat` = saturation
/// [0..4], `passes` = number of iterations [1..6].
pub fn punch(buf: &mut [f32], k: f32, sat: f32, passes: u32) {
    let k64   = k   as f64;
    let sat64 = sat as f64;
    let npix  = buf.len() / 4;

    // Pre-compute sigmoid lo/hi once (constant for a given k)
    let (lo, hi_minus_lo) = if k64 >= 1e-6 {
        let lo = sigmoid(0.0, k64);
        let hi = sigmoid(1.0, k64);
        (lo, hi - lo)
    } else {
        (0.0, 1.0) // unused when k < 1e-6
    };

    for _ in 0..passes {
        for px in 0..npix {
            let i = px * 4;

            // Read f32 → f64
            let r0 = buf[i    ] as f64;
            let g0 = buf[i + 1] as f64;
            let b0 = buf[i + 2] as f64;

            // --- Contrast ---
            let (r1, g1, b1) = if k64 < 1e-6 {
                (r0, g0, b0)
            } else {
                (
                    (sigmoid(r0, k64) - lo) / hi_minus_lo,
                    (sigmoid(g0, k64) - lo) / hi_minus_lo,
                    (sigmoid(b0, k64) - lo) / hi_minus_lo,
                )
            };

            // --- Saturation (luma from post-contrast RGB) ---
            let luma = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1;
            let r2   = (luma + sat64 * (r1 - luma)).clamp(0.0, 1.0);
            let g2   = (luma + sat64 * (g1 - luma)).clamp(0.0, 1.0);
            let b2   = (luma + sat64 * (b1 - luma)).clamp(0.0, 1.0);

            // Write back as f32 (clamp already applied)
            buf[i    ] = r2 as f32;
            buf[i + 1] = g2 as f32;
            buf[i + 2] = b2 as f32;
            // buf[i + 3] alpha unchanged
        }
    }
}

#[inline(always)]
fn sigmoid(x: f64, k: f64) -> f64 {
    1.0 / (1.0 + (-k * (x - 0.5)).exp())
}
