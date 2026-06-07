//! Generates reference PNGs for pixdiff validation against py/test_qbist.py.
//!
//! Writes (matching py/test_qbist.py names):
//!   test_qbist_seed0_os1.png
//!   test_qbist_seed0_os4.png
//!   test_qbist_seed1_os4.png
//!   test_qbist_seed42_os4.png
//!   test_qbist_confirm3500.png

use basecoat::qbist::{create_info, optimize, render};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

fn write_png(pixels: &[u8], size: usize, path: &str) {
    let file = File::create(path).unwrap_or_else(|e| panic!("create {path}: {e}"));
    let bw   = BufWriter::new(file);
    let mut enc = Encoder::new(bw, size as u32, size as u32);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("write header");
    writer.write_image_data(pixels).expect("write pixels");
    println!("wrote {path}");
}

fn run_case(name: &str, size: usize, seed: u64, os: usize) {
    let mut genome = create_info(seed);
    let (used_trans, used_reg) = optimize(&mut genome);
    let active_t = used_trans.iter().filter(|&&v| v).count();
    let active_r = used_reg.iter().filter(|&&v| v).count();
    println!(
        "  seed={seed} os={os} size={size}  active_transforms={active_t}  active_regs={active_r}  used_reg={used_reg:?}"
    );
    let pixels = render(&genome, &used_trans, &used_reg, size, size, os);
    write_png(&pixels, size, &format!("test_qbist_{name}.png"));
}

fn main() {
    run_case("seed0_os1",   256,  0,  1);
    run_case("seed0_os4",   256,  0,  4);
    run_case("seed1_os4",   256,  1,  4);
    run_case("seed42_os4",  256,  42, 4);
    run_case("confirm3500", 3500, 0,  4);
    println!("Done.");
}
