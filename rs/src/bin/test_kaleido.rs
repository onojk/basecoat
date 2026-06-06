//! Seam test + reference PNGs for pixdiff vs py/test_kaleido.py.
//! See spec/kaleido.md for algorithm and tolerance posture.

use basecoat::kaleido::kaleido;
use basecoat::layers::{linear_to_srgb_f64, Layer};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::f64::consts::TAU;
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

// ---------------------------------------------------------------------------
// PNG writer (sRGB — same encoding as py/test_kaleido.py)
// ---------------------------------------------------------------------------

fn write_png(buf: &[f32], w: u32, h: u32, path: &str) {
    let file = File::create(path).expect("create");
    let bw = BufWriter::new(file);
    let mut enc = Encoder::new(bw, w, h);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("header");

    let mut out = vec![0u8; (w * h * 4) as usize];
    for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
        let base = i * 4;
        chunk[0] = (linear_to_srgb_f64(buf[base    ] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[1] = (linear_to_srgb_f64(buf[base + 1] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[2] = (linear_to_srgb_f64(buf[base + 2] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[3] = (buf[base + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    writer.write_image_data(&out).expect("pixels");
    println!("wrote {path}");
}

// ---------------------------------------------------------------------------
// Source builders
// ---------------------------------------------------------------------------

fn gradient(size: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; size * size * 4];
    for row in 0..size {
        for col in 0..size {
            let i = (row * size + col) * 4;
            buf[i    ] = col as f32 / (size as f32 - 1.0);
            buf[i + 1] = row as f32 / (size as f32 - 1.0);
            buf[i + 2] = 0.5;
            buf[i + 3] = 1.0;
        }
    }
    buf
}

fn gradient_layer(size: usize) -> Layer {
    let buf = gradient(size);
    Layer { rgba: buf, width: size as u32, height: size as u32,
            mode: basecoat::layers::BlendMode::Normal,
            opacity: 1.0, visible: true, name: String::new() }
}

// ---------------------------------------------------------------------------
// Seam test helpers — scalar fold, mirrors kaleido.rs logic
// ---------------------------------------------------------------------------

fn fold_am(a: f64, wedge_w: f64) -> f64 {
    let k  = (a / wedge_w).floor();
    let am = (a - k * wedge_w).clamp(0.0, wedge_w);
    if (k as i64).rem_euclid(2) == 1 { wedge_w - am } else { am }
}

fn source_coord(x: f64, y: f64, cx: f64, cy: f64, wedge_w: f64, zoom: f64) -> (f64, f64) {
    let dx = x - cx;
    let dy = y - cy;
    let r  = dx.hypot(dy);
    let a  = dy.atan2(dx);
    let am = fold_am(a, wedge_w);
    (cx + (r / zoom) * am.cos(), cy + (r / zoom) * am.sin())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn test_seam_algorithm() {
    let tol = 1e-6_f64;
    let eps = 1e-8_f64;

    for &segments in &[2u32, 4, 6, 8, 12] {
        let wedge_w = TAU / segments as f64;
        let size    = 32usize;
        let cx      = (size as f64 - 1.0) / 2.0;
        let cy      = cx;
        let zoom    = 1.0_f64;

        for k in 0..segments {
            let seam = k as f64 * wedge_w;
            for &probe_r in &[4.0_f64, 8.0, 12.0] {
                let a1 = seam - eps;
                let a2 = seam + eps;
                let x1 = cx + probe_r * a1.cos();
                let y1 = cy + probe_r * a1.sin();
                let x2 = cx + probe_r * a2.cos();
                let y2 = cy + probe_r * a2.sin();

                let (sx1, sy1) = source_coord(x1, y1, cx, cy, wedge_w, zoom);
                let (sx2, sy2) = source_coord(x2, y2, cx, cy, wedge_w, zoom);

                assert!(
                    (sx1 - sx2).abs() < tol && (sy1 - sy2).abs() < tol,
                    "Seam k={k} seg={segments} r={probe_r}: \
                     sx {sx1:.9} vs {sx2:.9}, sy {sy1:.9} vs {sy2:.9}"
                );
            }
        }
    }
    println!("PASS  test_seam_algorithm");
}

fn test_seam_no_edge() {
    // Hard check: adjacent pixel pairs straddling the seam must differ < 0.02
    // per channel.  The edge-pixel-count along seam lines is informational only.
    // See spec/kaleido.md — the automated hard requirement is test_seam_algorithm.
    let size    = 64usize;
    let src     = gradient_layer(size);
    let out     = kaleido(&src, 6, 0.0, 1.0);

    let cx      = (size as f64 - 1.0) / 2.0;
    let cy      = cx;
    let wedge_w = TAU / 6.0;
    let min_r   = 3.0_f64;

    let mut max_diff   = 0.0_f64;
    let mut violations = 0usize;

    for k in 0..6u32 {
        let seam_angle = k as f64 * wedge_w;
        let cos_s = seam_angle.cos();
        let sin_s = seam_angle.sin();

        for y in 0..size {
            for x in 0..size {
                let r1 = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                if r1 < min_r { continue; }
                let cross1 = (y as f64 - cy) * cos_s - (x as f64 - cx) * sin_s;

                for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                    if nx >= size || ny >= size { continue; }
                    let r2 = ((nx as f64 - cx).powi(2) + (ny as f64 - cy).powi(2)).sqrt();
                    if r2 < min_r { continue; }
                    let cross2 = (ny as f64 - cy) * cos_s - (nx as f64 - cx) * sin_s;
                    if cross1 * cross2 >= 0.0 { continue; }

                    let i1 = (y * size + x) * 4;
                    let i2 = (ny * size + nx) * 4;
                    for c in 0..3 {
                        let d = (out.rgba[i1 + c] as f64 - out.rgba[i2 + c] as f64).abs();
                        if d > max_diff { max_diff = d; }
                        if d > 0.02 { violations += 1; }
                    }
                }
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Seam pixel-value diff > 0.02: {violations} channel violations, max={max_diff:.6}"
    );
    println!("PASS  test_seam_no_edge (max seam pair diff: {max_diff:.6})");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    test_seam_algorithm();
    test_seam_no_edge();

    // Reference PNGs for pixdiff
    let cases: &[(&str, usize, u32, f64, f64)] = &[
        ("grad32_s6_r0_z1",    32,  6,  0.0, 1.0),
        ("grad32_s8_r30_z1p5", 32,  8, 30.0, 1.5),
        ("grad64_s4_r0_z1",    64,  4,  0.0, 1.0),
    ];

    for &(name, size, segs, rot, zm) in cases {
        let src = gradient_layer(size);
        let out = kaleido(&src, segs, rot, zm);
        write_png(&out.rgba, size as u32, size as u32,
                  &format!("test_kaleido_{name}.png"));
    }

    println!("All kaleido tests passed.");
}
