use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const W: u32 = 3500;
const H: u32 = 3500;
// 600 PPI -> pixels per metre: round(600 / 0.0254) = 23622
const PPM: u32 = 23622;

fn main() {
    let path = "canvas.png";
    let file = File::create(path).expect("create canvas.png");
    let w = BufWriter::new(file);

    let mut enc = Encoder::new(w, W, H);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: PPM,
        yppu: PPM,
        unit: Unit::Meter,
    }));

    let mut writer = enc.write_header().expect("write header");

    let buf = vec![0u8; (W * H * 4) as usize];
    writer.write_image_data(&buf).expect("write pixels");

    println!("Wrote {path}  {W}x{H} RGBA  600 PPI ({PPM} px/m)");
}
