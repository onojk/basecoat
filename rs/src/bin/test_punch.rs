//! Generates test PNGs for pixdiff vs py/test_punch.py.

use basecoat::layers::{linear_to_srgb_f64, Layer};
use basecoat::plasma::{apply_plasma, H as PLASMA_H, W as PLASMA_W};
use basecoat::punch::punch;
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const S: u32   = 64;
const BIG: u32 = 3500;
const PPM: u32 = 23622;

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

/// R = col/(size-1), G = row/(size-1), B = 0.5, A = 1.0
fn gradient(size: u32) -> Vec<f32> {
    let n = size as usize;
    let mut buf = vec![0.0f32; n * n * 4];
    for row in 0..n {
        for col in 0..n {
            let i = (row * n + col) * 4;
            buf[i    ] = col as f32 / (n as f32 - 1.0);
            buf[i + 1] = row as f32 / (n as f32 - 1.0);
            buf[i + 2] = 0.5;
            buf[i + 3] = 1.0;
        }
    }
    buf
}

/// Top-left SxS crop of plasma (seed=0, turb=1.0)
fn plasma_crop(size: u32) -> Vec<f32> {
    // generate full plasma then crop
    let mut full = Layer::new(PLASMA_W as u32, PLASMA_H as u32, [0.0f32; 4]);
    apply_plasma(&mut full.rgba, 0, 1.0);
    let s = size as usize;
    let mut buf = vec![0.0f32; s * s * 4];
    for row in 0..s {
        for col in 0..s {
            let src = (row * PLASMA_W + col) * 4;
            let dst = (row * s + col) * 4;
            buf[dst    ] = full.rgba[src    ];
            buf[dst + 1] = full.rgba[src + 1];
            buf[dst + 2] = full.rgba[src + 2];
            buf[dst + 3] = full.rgba[src + 3];
        }
    }
    buf
}

fn run_case(buf: &[f32], k: f32, sat: f32, passes: u32, w: u32, h: u32, path: &str) {
    let mut b = buf.to_vec();
    punch(&mut b, k, sat, passes);
    write_png(&b, w, h, path);
}

fn main() {
    let grad = gradient(S);

    run_case(&grad, 9.0, 4.0, 6, S, S, "test_punch_grad_default.png");
    run_case(&grad, 0.0, 1.0, 1, S, S, "test_punch_grad_identity.png");
    run_case(&grad, 9.0, 0.0, 1, S, S, "test_punch_grad_grayscale.png");
    run_case(&grad, 9.0, 4.0, 1, S, S, "test_punch_grad_1pass.png");

    eprintln!("generating plasma crop...");
    let plasma = plasma_crop(S);
    run_case(&plasma, 9.0, 4.0, 6, S, S, "test_punch_plasma_default.png");

    eprintln!("generating 3500x3500 confirming run...");
    let big = gradient(BIG);
    run_case(&big, 9.0, 4.0, 6, BIG, BIG, "test_punch_3500.png");
}
