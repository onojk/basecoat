//! Band generator — concentric outward fill from a seed mask.
//! See spec/bands.md.

/// One generated band: which pixels it covers and what colour.
pub struct Band {
    pub mask: Vec<bool>, // W*H, row-major; true = this band's pixel
    pub white: bool,     // true = white (#fff), false = black (#000)
}

/// Map growth percentage [0,100] to thickness in pixels [1,40].
/// Uses round-half-up so both Python and Rust agree on every integer growth%.
pub fn growth_to_thickness(growth_pct: f32) -> usize {
    let x = 1.0_f32 + (growth_pct / 100.0) * 39.0;
    (x + 0.5).floor() as usize
}

/// Generate bands outward from `seed` (boolean W×H mask, row-major).
/// Returns bands ordered **innermost first** (index 0 = band closest to seed,
/// white). Thickness `thickness_px` is the Chebyshev dilation radius per
/// iteration. Stops when coverage ≥ fill_pct% or the canvas saturates.
pub fn generate_bands(
    seed: &[bool],
    width: usize,
    height: usize,
    thickness_px: usize,
    fill_pct: f32,
) -> Vec<Band> {
    let total = width * height;
    let mut coverage: Vec<bool> = seed.to_vec();
    let mut coverage_count: usize = coverage.iter().filter(|&&b| b).count();
    let mut bands: Vec<Band> = Vec::new();
    let fill_threshold = (fill_pct / 100.0 * total as f32).ceil() as usize;

    loop {
        let dilated  = dilate_chebyshev(&coverage, width, height, thickness_px);
        let new_ring: Vec<bool> = dilated.iter().zip(coverage.iter())
            .map(|(&d, &c)| d && !c)
            .collect();

        let new_count = new_ring.iter().filter(|&&b| b).count();
        if new_count == 0 {
            break; // stop condition 2: canvas saturated
        }

        let is_white = bands.len().is_multiple_of(2); // white first
        bands.push(Band { mask: new_ring.clone(), white: is_white });

        for (c, n) in coverage.iter_mut().zip(new_ring.iter()) {
            if *n { *c = true; }
        }
        coverage_count += new_count;

        if coverage_count >= fill_threshold {
            break; // stop condition 1: fill% reached
        }
    }

    bands
}

// ---------------------------------------------------------------------------
// Chebyshev (box) dilation — two-pass 1D sliding window, O(W·H) per call.
// Matches py/bands.py's dilate_chebyshev exactly.
// ---------------------------------------------------------------------------

pub fn dilate_chebyshev(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }

    // ---- Horizontal pass ----
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        let base = y * w;
        // Initialize count for x=0: window covers [0 .. min(r, w-1)]
        let mut count: usize = (0..=r.min(w.saturating_sub(1)))
            .filter(|&nx| mask[base + nx])
            .count();
        tmp[base] = count > 0;
        for x in 1..w {
            // Right edge enters window
            if x + r < w && mask[base + x + r] { count += 1; }
            // Left edge leaves window
            if x > r && mask[base + x - r - 1] { count -= 1; }
            tmp[base + x] = count > 0;
        }
    }

    // ---- Vertical pass ----
    let mut result = vec![false; w * h];
    for x in 0..w {
        // Initialize count for y=0: window covers [0 .. min(r, h-1)]
        let mut count: usize = (0..=r.min(h.saturating_sub(1)))
            .filter(|&ny| tmp[ny * w + x])
            .count();
        result[x] = count > 0;
        for y in 1..h {
            if y + r < h && tmp[(y + r) * w + x] { count += 1; }
            if y > r     && tmp[(y - r - 1) * w + x] { count -= 1; }
            result[y * w + x] = count > 0;
        }
    }

    result
}
