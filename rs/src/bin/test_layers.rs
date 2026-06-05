//! Outputs test PNGs for pixdiff validation against py/test_layers.py.

use basecoat::layers::*;
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

fn write_png(layer: &Layer, path: &str) {
    let (w, h) = (layer.width, layer.height);
    let file = File::create(path).expect("create png");
    let bw = BufWriter::new(file);
    let mut enc = Encoder::new(bw, w, h);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("header");

    // Cast f32 → f64 before sRGB conversion, matching Python's astype(np.float64) path.
    // Alpha passes through unchanged (no gamma).
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for (i, chunk) in buf.chunks_exact_mut(4).enumerate() {
        let base = i * 4;
        chunk[0] = (linear_to_srgb_f64(layer.rgba[base    ] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[1] = (linear_to_srgb_f64(layer.rgba[base + 1] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[2] = (linear_to_srgb_f64(layer.rgba[base + 2] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        chunk[3] = (layer.rgba[base + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    writer.write_image_data(&buf).expect("pixels");
    println!("wrote {path}");
}

fn main() {
    let s = 64u32;

    // --- test_normal: red opaque over blue opaque → pure red ---
    {
        let bot = Layer::new(s, s, [0.0, 0.0, 1.0, 1.0]);
        let top = Layer::new(s, s, [1.0, 0.0, 0.0, 1.0]);
        write_png(&composite(&[bot, top]), "test_normal.png");
    }

    // --- test_multiply ---
    {
        let bot = Layer::new(s, s, [0.0, 0.0, 0.5, 1.0]);
        let mut top = Layer::new(s, s, [0.8, 0.0, 0.0, 1.0]);
        top.mode = BlendMode::Multiply;
        write_png(&composite(&[bot, top]), "test_multiply.png");
    }

    // --- test_screen ---
    {
        let bot = Layer::new(s, s, [0.0, 0.0, 0.5, 1.0]);
        let mut top = Layer::new(s, s, [0.8, 0.0, 0.0, 1.0]);
        top.mode = BlendMode::Screen;
        write_png(&composite(&[bot, top]), "test_screen.png");
    }

    // --- test_overlay ---
    {
        let bot = Layer::new(s, s, [0.0, 0.0, 0.3, 1.0]);
        let mut top = Layer::new(s, s, [0.8, 0.0, 0.0, 1.0]);
        top.mode = BlendMode::Overlay;
        write_png(&composite(&[bot, top]), "test_overlay.png");
    }

    // --- test_difference ---
    {
        let bot = Layer::new(s, s, [0.5, 0.5, 0.5, 1.0]);
        let mut top = Layer::new(s, s, [0.8, 0.2, 0.3, 1.0]);
        top.mode = BlendMode::Difference;
        write_png(&composite(&[bot, top]), "test_difference.png");
    }

    // --- test_opacity: semi-transparent top over solid bottom ---
    {
        let bot = Layer::new(s, s, [0.0, 0.8, 0.0, 1.0]);
        let mut top = Layer::new(s, s, [1.0, 0.0, 0.0, 1.0]);
        top.opacity = 0.5;
        write_png(&composite(&[bot, top]), "test_opacity.png");
    }

    // --- test_alpha: two translucent layers ---
    {
        let bot = Layer::new(s, s, [0.0, 0.0, 1.0, 0.8]);
        let top = Layer::new(s, s, [1.0, 0.0, 0.0, 0.6]);
        write_png(&composite(&[bot, top]), "test_alpha.png");
    }

    // --- test_invisible: invisible layer is skipped ---
    {
        let bot = Layer::new(s, s, [0.0, 0.5, 0.0, 1.0]);
        let mut mid = Layer::new(s, s, [1.0, 0.0, 0.0, 1.0]);
        mid.visible = false;
        let top = Layer::new(s, s, [0.0, 0.0, 0.8, 0.5]);
        write_png(&composite(&[bot, mid, top]), "test_invisible.png");
    }

    // --- test_flatten: flatten then composite again ---
    {
        let mut stack = Stack::new();
        let a = Layer::new(s, s, [0.2, 0.4, 0.6, 1.0]);
        let mut b = Layer::new(s, s, [0.8, 0.1, 0.1, 0.7]);
        b.mode = BlendMode::Screen;
        stack.add(a).unwrap();
        stack.add(b).unwrap();
        stack.flatten_visible();
        write_png(&stack.composite(), "test_flatten.png");
    }

    // --- test_undo: fill then undo → composite original color ---
    {
        let mut stack = Stack::new();
        let a = Layer::new(s, s, [0.3, 0.3, 0.3, 1.0]);
        stack.add(a).unwrap();
        stack.fill(0, [1.0, 0.0, 0.0, 1.0]);
        stack.undo();
        write_png(&stack.composite(), "test_undo.png");
    }

    // --- test_3500: full-size overlay composite (confirming run) ---
    {
        let big = 3500u32;
        let bot = Layer::new(big, big, [0.18, 0.18, 0.18, 1.0]);
        let mut top = Layer::new(big, big, [0.8, 0.4, 0.1, 0.5]);
        top.mode = BlendMode::Overlay;
        write_png(&composite(&[bot, top]), "test_3500.png");
    }
}
