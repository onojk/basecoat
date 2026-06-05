//! Generates edge PNGs for pixdiff vs py/test_edge.py.

use basecoat::edge::edge;
use basecoat::plasma::{apply_plasma, H as PLASMA_H, W as PLASMA_W};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const S: usize   = 64;
const BIG: usize = 3500;
const PPM: u32   = 23622;

fn write_png(buf: &[f32], w: u32, h: u32, path: &str) {
    // Edge output: 0.0 or 1.0 only; direct u8 encode (no sRGB needed for B/W/transparent).
    let file = File::create(path).expect("create");
    let bw   = BufWriter::new(file);
    let mut enc = Encoder::new(bw, w, h);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("header");

    let npix = (w * h) as usize;
    let mut out = vec![0u8; npix * 4];
    let mut edge_count = 0usize;
    for i in 0..npix {
        let base = i * 4;
        // RGB are 0.0 (black); alpha is 0.0 or 1.0
        out[base + 3] = (buf[base + 3] * 255.0 + 0.5) as u8;
        if buf[base + 3] > 0.5 { edge_count += 1; }
    }
    writer.write_image_data(&out).expect("pixels");
    println!("wrote {path}  edge_px={edge_count}");
}

fn run(buf: &[f32], w: usize, h: usize, aggr: f32, lw: usize, path: &str) {
    let result = edge(buf, w, h, aggr, lw);
    write_png(&result, w as u32, h as u32, path);
}

fn solid(size: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
    let mut buf = vec![0.0f32; size * size * 4];
    for i in 0..size * size {
        buf[i*4    ] = r;
        buf[i*4 + 1] = g;
        buf[i*4 + 2] = b;
        buf[i*4 + 3] = 1.0;
    }
    buf
}

fn split(size: usize) -> Vec<f32> {
    let mut buf = solid(size, 0.2, 0.4, 0.6);
    for row in 0..size {
        for col in size/2..size {
            let base = (row * size + col) * 4;
            buf[base    ] = 0.8;
            buf[base + 1] = 0.2;
            buf[base + 2] = 0.3;
        }
    }
    buf
}

fn quad(size: usize) -> Vec<f32> {
    let h2 = size / 2;
    let colors: [[f32; 3]; 4] = [
        [0.1, 0.9, 0.1],
        [0.9, 0.1, 0.1],
        [0.1, 0.1, 0.9],
        [0.9, 0.9, 0.9],
    ];
    let mut buf = vec![0.0f32; size * size * 4];
    for row in 0..size {
        for col in 0..size {
            let qi = (if row >= h2 { 2 } else { 0 }) + (if col >= h2 { 1 } else { 0 });
            let base = (row * size + col) * 4;
            buf[base    ] = colors[qi][0];
            buf[base + 1] = colors[qi][1];
            buf[base + 2] = colors[qi][2];
            buf[base + 3] = 1.0;
        }
    }
    buf
}

fn plasma_crop(size: usize) -> Vec<f32> {
    use basecoat::layers::Layer;
    let mut full = Layer::new(PLASMA_W as u32, PLASMA_H as u32, [0.0f32; 4]);
    apply_plasma(&mut full.rgba, 0, 1.0);
    let mut buf = vec![0.0f32; size * size * 4];
    for row in 0..size {
        for col in 0..size {
            let src = (row * PLASMA_W + col) * 4;
            let dst = (row * size   + col) * 4;
            buf[dst    ] = full.rgba[src    ];
            buf[dst + 1] = full.rgba[src + 1];
            buf[dst + 2] = full.rgba[src + 2];
            buf[dst + 3] = full.rgba[src + 3];
        }
    }
    buf
}

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

fn main() {
    // 64×64 split
    run(&split(S), S, S, 50.0, 1, "test_edge_split_a50_w1.png");
    run(&split(S), S, S, 50.0, 3, "test_edge_split_a50_w3.png");
    run(&split(S), S, S, 50.0, 7, "test_edge_split_a50_w7.png");

    // 64×64 quad
    run(&quad(S), S, S,  50.0, 3, "test_edge_quad_a50_w3.png");
    run(&quad(S), S, S,   0.0, 3, "test_edge_quad_a0_w3.png");
    run(&quad(S), S, S, 100.0, 3, "test_edge_quad_a100_w3.png");

    // 64×64 plasma crop
    eprintln!("generating plasma crop...");
    run(&plasma_crop(S), S, S, 50.0, 3, "test_edge_plasma_a50_w3.png");

    // 3500×3500 confirming run (gradient — fast, deterministic)
    eprintln!("generating 3500×3500 confirming run...");
    run(&gradient(BIG), BIG, BIG, 50.0, 3, "test_edge_confirm3500.png");
}
