//! Generates composite PNGs of band generation for pixdiff vs py/test_bands.py.

use basecoat::bands::{generate_bands, growth_to_thickness};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

fn write_composite(seed: &[bool], bands: &[basecoat::bands::Band], w: u32, h: u32, path: &str) {
    let mut rgba = vec![0u8; (w * h * 4) as usize];

    // Seed: black opaque
    for (i, &s) in seed.iter().enumerate() {
        if s {
            let base = i * 4;
            rgba[base    ] = 0;
            rgba[base + 1] = 0;
            rgba[base + 2] = 0;
            rgba[base + 3] = 255;
        }
    }

    // Bands (non-overlapping, any draw order is fine)
    for band in bands {
        let v = if band.white { 255u8 } else { 0u8 };
        for (i, &m) in band.mask.iter().enumerate() {
            if m {
                let base = i * 4;
                rgba[base    ] = v;
                rgba[base + 1] = v;
                rgba[base + 2] = v;
                rgba[base + 3] = 255;
            }
        }
    }

    let file = File::create(path).expect("create");
    let bw = BufWriter::new(file);
    let mut enc = Encoder::new(bw, w, h);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("header");
    writer.write_image_data(&rgba).expect("pixels");
    println!("wrote {path}  bands={}", bands.len());
}

fn dot_seed(size: usize) -> Vec<bool> {
    let mut s = vec![false; size * size];
    s[(size / 2) * size + (size / 2)] = true;
    s
}

fn diag_seed(size: usize) -> Vec<bool> {
    let mut s = vec![false; size * size];
    for i in 0..size { s[i * size + i] = true; }
    s
}

fn run(seed: &[bool], w: usize, h: usize, growth: f32, fill: f32, path: &str) {
    let t     = growth_to_thickness(growth);
    let bands = generate_bands(seed, w, h, t, fill);
    write_composite(seed, &bands, w as u32, h as u32, path);
}

fn main() {
    let s = 64usize;
    let c = 256usize;

    run(&dot_seed(s),  s, s,  30.0, 90.0, "test_bands_dot_g30.png");
    run(&dot_seed(s),  s, s,  10.0, 90.0, "test_bands_dot_g10.png");
    run(&dot_seed(s),  s, s,  50.0, 90.0, "test_bands_dot_g50.png");
    run(&diag_seed(s), s, s,  30.0, 90.0, "test_bands_diag_g30.png");
    run(&dot_seed(c),  c, c,  30.0, 80.0, "test_bands_confirm256.png");
}
