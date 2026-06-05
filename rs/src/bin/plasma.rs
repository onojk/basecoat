use png::{BitDepth, ColorType, Encoder, Unit};
use std::fs::File;
use std::io::BufWriter;

const W: usize = 3500;
const H: usize = 3500;
const N: usize = 4097; // 2^12 + 1
const PPM: u32 = 23622;

// --- PRNG: xorshift64* ---------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f64 {
        let mut s = self.state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.state = s;
        let r = s.wrapping_mul(0x2545F4914F6CDD1D);
        (r >> 11) as f64 / 9007199254740992.0
    }
}

// --- Diamond-square -------------------------------------------------------

fn diamond_square(rng: &mut Rng, turbulence: f64) -> Vec<f64> {
    let mut grid = vec![0.0f64; N * N];

    macro_rules! g {
        ($r:expr, $c:expr) => {
            grid[$r * N + $c]
        };
    }

    // Seed corners row-major: (0,0), (0,N-1), (N-1,0), (N-1,N-1)
    g![0,     0    ] = rng.unit();
    g![0,     N - 1] = rng.unit();
    g![N - 1, 0    ] = rng.unit();
    g![N - 1, N - 1] = rng.unit();

    let mut scale = turbulence;
    let mut step = N - 1; // 4096

    while step > 1 {
        let half = step / 2;

        // Diamond step — row-major over square centers
        let mut r = 0;
        while r < N - 1 {
            let mut c = 0;
            while c < N - 1 {
                let mean = (g![r, c] + g![r, c + step] +
                            g![r + step, c] + g![r + step, c + step]) * 0.25;
                let disp = (rng.unit() * 2.0 - 1.0) * scale;
                g![r + half, c + half] = mean + disp;
                c += step;
            }
            r += step;
        }

        // Square step — row-major over diamond edge midpoints.
        // Even row-bands (r/half even) start at column=half;
        // odd row-bands start at column=0.  Both step by `step`.
        let mut r = 0usize;
        while r < N {
            let start = if (r / half) % 2 == 0 { half } else { 0 };
            let mut c = start;
            while c < N {
                let mut total = 0.0f64;
                let mut count = 0u32;
                for (dr, dc) in [
                    (-(half as isize), 0isize),
                    (half as isize,    0isize),
                    (0isize,           -(half as isize)),
                    (0isize,           half as isize),
                ] {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < N as isize && nc >= 0 && nc < N as isize {
                        total += grid[nr as usize * N + nc as usize];
                        count += 1;
                    }
                }
                let disp = (rng.unit() * 2.0 - 1.0) * scale;
                g![r, c] = total / count as f64 + disp;
                c += step;
            }
            r += half;
        }

        scale *= 0.5;
        step = half;
    }

    grid.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
    grid
}

// --- sRGB encode ----------------------------------------------------------

fn linear_to_srgb(v: f64) -> u8 {
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

// --- Main -----------------------------------------------------------------

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

    let seed_r = seed;
    let seed_g = seed ^ 0x9E3779B97F4A7C15u64;
    let seed_b = seed ^ 0xD1B54A32D192ED03u64;

    let r_grid = diamond_square(&mut Rng::new(seed_r), turbulence);
    let g_grid = diamond_square(&mut Rng::new(seed_g), turbulence);
    let b_grid = diamond_square(&mut Rng::new(seed_b), turbulence);

    let mut buf = vec![0u8; W * H * 4];
    for row in 0..H {
        for col in 0..W {
            let idx = row * N + col;
            let out = (row * W + col) * 4;
            buf[out    ] = linear_to_srgb(r_grid[idx]);
            buf[out + 1] = linear_to_srgb(g_grid[idx]);
            buf[out + 2] = linear_to_srgb(b_grid[idx]);
            buf[out + 3] = 255;
        }
    }

    let path = "plasma.png";
    let file = File::create(path).expect("create plasma.png");
    let w = BufWriter::new(file);
    let mut enc = Encoder::new(w, W as u32, H as u32);
    enc.set_color(ColorType::Rgba);
    enc.set_depth(BitDepth::Eight);
    enc.set_pixel_dims(Some(png::PixelDimensions {
        xppu: PPM,
        yppu: PPM,
        unit: Unit::Meter,
    }));
    let mut writer = enc.write_header().expect("write header");
    writer.write_image_data(&buf).expect("write pixels");

    println!("Wrote {path}  seed={seed}  turbulence={turbulence}");
}
