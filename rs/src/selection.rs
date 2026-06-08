//! Global select-by-color tool.
//! Returns a per-pixel coverage mask (0.0 = unselected, 1.0 = fully selected,
//! fractional = feathered soft edge). Ignores alpha; uses linear-RGB Euclidean distance.

// sqrt(3) — max Euclidean distance in unit RGB cube
const MAX_DIST: f64 = 1.7320508075688772;

/// Build a W×H coverage mask.
///
/// `aggr_pct` 0..=100: 100 = loosest (selects all), 0 = strictest (exact match only).
/// `feather_pct` 0..=100: 0 = hard edge, 100 = maximum soft falloff.
/// Input `rgba` is linear-light straight-alpha (layer native format).
#[allow(clippy::too_many_arguments)]
pub fn select_by_color(
    rgba:        &[f32],
    w:           usize,
    h:           usize,
    pr:          f32,
    pg:          f32,
    pb:          f32,
    aggr_pct:    f32,
    feather_pct: f32,
) -> Vec<f32> {
    let tol     = (aggr_pct    as f64 / 100.0) * MAX_DIST;
    let feather = (feather_pct as f64 / 100.0) * MAX_DIST;
    let pr = pr as f64;
    let pg = pg as f64;
    let pb = pb as f64;

    (0..w * h)
        .map(|i| {
            let base = i * 4;
            let dr = rgba[base    ] as f64 - pr;
            let dg = rgba[base + 1] as f64 - pg;
            let db = rgba[base + 2] as f64 - pb;
            let d  = (dr * dr + dg * dg + db * db).sqrt();
            if d <= tol {
                1.0_f32
            } else if feather == 0.0 || d >= tol + feather {
                0.0_f32
            } else {
                (1.0 - (d - tol) / feather) as f32
            }
        })
        .collect()
}

/// Grow `base` mask by unioning with `pick`: base[i] = max(base[i], pick[i]).
pub fn union_masks(base: &mut [f32], pick: &[f32]) {
    for (b, &p) in base.iter_mut().zip(pick.iter()) {
        if p > *b { *b = p; }
    }
}

/// Fraction of the mask that is at least partially selected (weighted by coverage), as a %.
pub fn mask_coverage_pct(mask: &[f32]) -> f64 {
    if mask.is_empty() { return 0.0; }
    let sum: f64 = mask.iter().map(|&v| v as f64).sum();
    sum / mask.len() as f64 * 100.0
}
