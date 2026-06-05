//! Headless GUI export test: single plasma layer, seed=0, turb=1.0, Normal.
//! Exercises the same export_png path the GUI uses.
//! Run: cargo run --bin export_test && pixdiff rs/export_test.png rs/plasma.png

use basecoat::layers::*;
use basecoat::plasma::{apply_plasma, H as IMG_H, W as IMG_W};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::io::BufWriter;

const W: u32 = IMG_W as u32;
const H: u32 = IMG_H as u32;
const PPM: u32 = 23622;

fn main() {
    // Build the same stack the GUI starts with, then apply plasma
    let mut stack = Stack::new();
    let mut base = Layer::new(W, H, [0.0f32; 4]);
    apply_plasma(&mut base.rgba, 0, 1.0);
    stack.add(base).unwrap();

    // composite() — same code path the GUI Export button calls
    let comp = stack.composite();

    // encode — same srgb_enc as in basecoat_gui.rs
    let path = "export_test.png";
    let file = std::fs::File::create(path).unwrap();
    let bw = BufWriter::new(file);
    let mut enc = Encoder::new(bw, W, H);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: png::Unit::Meter }));
    let mut writer = enc.write_header().unwrap();

    let mut buf = vec![0u8; (W * H * 4) as usize];
    for i in (0..comp.rgba.len()).step_by(4) {
        buf[i    ] = (linear_to_srgb_f64(comp.rgba[i    ] as f64).clamp(0.0,1.0) * 255.0 + 0.5) as u8;
        buf[i + 1] = (linear_to_srgb_f64(comp.rgba[i + 1] as f64).clamp(0.0,1.0) * 255.0 + 0.5) as u8;
        buf[i + 2] = (linear_to_srgb_f64(comp.rgba[i + 2] as f64).clamp(0.0,1.0) * 255.0 + 0.5) as u8;
        buf[i + 3] = (comp.rgba[i + 3].clamp(0.0,1.0) * 255.0 + 0.5) as u8;
    }
    writer.write_image_data(&buf).unwrap();
    println!("Wrote {path} — pixdiff against rs/plasma.png to verify");
}
