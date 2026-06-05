//! Edge technique — posterize-then-outline. See spec/edge.md.
//! Reuses dilate_chebyshev from bands::dilate_chebyshev.

use crate::bands::dilate_chebyshev;

/// Map aggressiveness [0,100] to posterize level count N [3,32] (round-half-up).
pub fn aggr_to_n(aggr: f32) -> i32 {
    (3.0_f32 + (aggr / 100.0) * 29.0 + 0.5).floor() as i32
}

/// Posterize one f32 channel value [0,1] to bucket index 0..N-1 (round-half-up).
#[inline]
fn bucket(c: f32, n: i32) -> i32 {
    let x = c as f64 * (n - 1) as f64 + 0.5;
    (x.floor() as i32).clamp(0, n - 1)
}

/// Compute edge mask and return a new RGBA f32 layer buffer (W×H×4, row-major).
/// Edge pixels = opaque black (0,0,0,1). All others = transparent (0,0,0,0).
/// The source buffer is not modified.
///
/// `rgba` — source layer, f32 linear, W×H×4 row-major.
/// `aggr` — aggressiveness 0..100 (maps to N via `aggr_to_n`).
/// `line_width` — 1..7; dilation radius = line_width / 2.
pub fn edge(rgba: &[f32], width: usize, height: usize, aggr: f32, line_width: usize) -> Vec<f32> {
    let n    = aggr_to_n(aggr);
    let npix = width * height;

    // Step 1: posterize each pixel to (br, bg, bb) bucket triple
    let mut bucks: Vec<[i32; 3]> = Vec::with_capacity(npix);
    for i in 0..npix {
        let base = i * 4;
        bucks.push([
            bucket(rgba[base    ], n),
            bucket(rgba[base + 1], n),
            bucket(rgba[base + 2], n),
        ]);
    }

    // Step 2: 4-neighbour boundary detection (no wrap; border = same)
    let mut mask = vec![false; npix];
    for y in 0..height {
        for x in 0..width {
            let me = &bucks[y * width + x];
            let edge =
                (y > 0           && bucks[(y-1)*width + x    ] != *me) ||
                (y + 1 < height  && bucks[(y+1)*width + x    ] != *me) ||
                (x > 0           && bucks[y*width     + x - 1] != *me) ||
                (x + 1 < width   && bucks[y*width     + x + 1] != *me);
            if edge { mask[y * width + x] = true; }
        }
    }

    // Step 3: dilation (reuse Chebyshev from bands)
    let r = line_width / 2;
    let dilated = if r > 0 {
        dilate_chebyshev(&mask, width, height, r)
    } else {
        mask
    };

    // Step 4: build output layer (opaque black on edge, transparent elsewhere)
    let mut out = vec![0.0f32; npix * 4];
    for i in 0..npix {
        if dilated[i] {
            // out[i*4 + 0..2] stay 0.0 (black)
            out[i * 4 + 3] = 1.0; // opaque
        }
    }
    out
}
