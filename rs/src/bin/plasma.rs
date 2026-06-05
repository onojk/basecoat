use basecoat::layers::{linear_to_srgb_f64, Layer};
use basecoat::plasma::{apply_plasma, H, W};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

fn main() {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args
        .next()
        .as_deref()
        .and_then(|s| {
            if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                u64::from_str_radix(h, 16).ok()
            } else {
                s.parse().ok()
            }
        })
        .unwrap_or(0);
    let turbulence: f64 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let mut layer = Layer::new(W as u32, H as u32, [0.0; 4]);
    apply_plasma(&mut layer.rgba, seed, turbulence);

    let path = "plasma.png";
    let file = File::create(path).expect("create plasma.png");
    let w    = BufWriter::new(file);
    let mut enc = Encoder::new(w, W as u32, H as u32);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: PPM,
        yppu: PPM,
        unit: Unit::Meter,
    }));
    let mut writer = enc.write_header().expect("write header");

    let mut buf = vec![0u8; W * H * 4];
    for i in (0..layer.rgba.len()).step_by(4) {
        buf[i    ] = (linear_to_srgb_f64(layer.rgba[i    ] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf[i + 1] = (linear_to_srgb_f64(layer.rgba[i + 1] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf[i + 2] = (linear_to_srgb_f64(layer.rgba[i + 2] as f64).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf[i + 3] = (layer.rgba[i + 3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    writer.write_image_data(&buf).expect("write pixels");
    println!("Wrote {path}  seed={seed}  turbulence={turbulence}");
}
