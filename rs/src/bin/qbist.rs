use basecoat::qbist::{create_info, optimize, render};
use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const PPM: u32 = 23622;

fn main() {
    let mut args = std::env::args().skip(1);

    let seed: u64 = args
        .next()
        .as_deref()
        .map(parse_seed)
        .unwrap_or(0);
    let os: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let size: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3500);

    let mut genome = create_info(seed);
    let (used_trans, used_reg) = optimize(&mut genome);

    let active_t = used_trans.iter().filter(|&&v| v).count();
    let active_r = used_reg.iter().filter(|&&v| v).count();
    println!(
        "seed={seed}  active_transforms={active_t}  active_regs={active_r}  used_reg={used_reg:?}"
    );

    let pixels = render(&genome, &used_trans, &used_reg, size, size, os);

    let path = "qbist.png";
    let file  = File::create(path).expect("create qbist.png");
    let bw    = BufWriter::new(file);
    let mut enc = Encoder::new(bw, size as u32, size as u32);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions { xppu: PPM, yppu: PPM, unit: Unit::Meter }));
    let mut writer = enc.write_header().expect("write header");
    writer.write_image_data(&pixels).expect("write pixels");
    println!("Wrote {path}  size={size}  os={os}");
}

fn parse_seed(s: &str) -> u64 {
    if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).unwrap_or(0)
    } else {
        s.parse().unwrap_or(0)
    }
}
