//! Mirror-tile (Quad) transform.
//!
//! Area-average downscales the source to one cell (cw×ch), then tiles it n×n
//! with alternating mirror flips so every seam is seamless.

/// Mirror-tile `src` (linear f32 RGBA, w×h) into an n×n grid.
///
/// Returns a new w×h buffer.  Cell size `cw = w/n`, `ch = h/n` (integer
/// division, min 1).  Remainder columns/rows are filled by clamping `lx`/`ly`
/// to `cw-1`/`ch-1`, so the output has no transparent strips.
pub fn quad(src: &[f32], w: usize, h: usize, n: usize) -> Vec<f32> {
    assert!(n >= 1, "n must be >= 1");
    let cw = (w / n).max(1);
    let ch = (h / n).max(1);

    // Area-average downscale to cw×ch in f64 (physically correct, no gamma).
    let mut cell = vec![0.0f64; cw * ch * 4];
    for ly in 0..ch {
        let y_lo = ly * h / ch;
        let y_hi = ((ly + 1) * h / ch).max(y_lo + 1).min(h);
        for lx in 0..cw {
            let x_lo = lx * w / cw;
            let x_hi = ((lx + 1) * w / cw).max(x_lo + 1).min(w);
            let count = ((y_hi - y_lo) * (x_hi - x_lo)) as f64;
            let (mut sr, mut sg, mut sb, mut sa) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for y in y_lo..y_hi {
                for x in x_lo..x_hi {
                    let i = (y * w + x) * 4;
                    sr += src[i    ] as f64;
                    sg += src[i + 1] as f64;
                    sb += src[i + 2] as f64;
                    sa += src[i + 3] as f64;
                }
            }
            let base = (ly * cw + lx) * 4;
            cell[base    ] = sr / count;
            cell[base + 1] = sg / count;
            cell[base + 2] = sb / count;
            cell[base + 3] = sa / count;
        }
    }

    // Mirror-tile into w×h output.
    let mut out = vec![0.0f32; w * h * 4];
    for y in 0..h {
        let gy      = (y / ch).min(n - 1);
        let ly_raw  = (y - gy * ch).min(ch - 1);
        let ly      = if gy % 2 == 1 { ch - 1 - ly_raw } else { ly_raw };
        for x in 0..w {
            let gx      = (x / cw).min(n - 1);
            let lx_raw  = (x - gx * cw).min(cw - 1);
            let lx      = if gx % 2 == 1 { cw - 1 - lx_raw } else { lx_raw };
            let ci = (ly * cw + lx) * 4;
            let oi = (y * w + x) * 4;
            out[oi    ] = cell[ci    ] as f32;
            out[oi + 1] = cell[ci + 1] as f32;
            out[oi + 2] = cell[ci + 2] as f32;
            out[oi + 3] = cell[ci + 3] as f32;
        }
    }

    out
}
