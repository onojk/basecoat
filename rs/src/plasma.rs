//! Diamond-square plasma generator — see spec/plasma.md.
//! Shared between the headless binary and the GUI.

pub const GRID: usize = 4097; // 2^12 + 1
pub const W: usize = 3500;
pub const H: usize = 3500;

// --- PRNG: xorshift64* ---------------------------------------------------

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn unit(&mut self) -> f64 {
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
    let n = GRID;
    let mut grid = vec![0.0f64; n * n];

    macro_rules! g {
        ($r:expr, $c:expr) => { grid[$r * n + $c] };
    }

    g![0,     0    ] = rng.unit();
    g![0,     n - 1] = rng.unit();
    g![n - 1, 0    ] = rng.unit();
    g![n - 1, n - 1] = rng.unit();

    let mut scale = turbulence;
    let mut step  = n - 1;

    while step > 1 {
        let half = step / 2;

        // Diamond step — row-major over centers
        let mut r = 0;
        while r < n - 1 {
            let mut c = 0;
            while c < n - 1 {
                let mean = (g![r, c] + g![r, c + step]
                          + g![r + step, c] + g![r + step, c + step]) * 0.25;
                g![r + half, c + half] = mean + (rng.unit() * 2.0 - 1.0) * scale;
                c += step;
            }
            r += step;
        }

        // Square step — row-major over edge midpoints
        let mut r = 0usize;
        while r < n {
            let start = if (r / half).is_multiple_of(2) { half } else { 0 };
            let mut c = start;
            while c < n {
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
                    if nr >= 0 && nr < n as isize && nc >= 0 && nc < n as isize {
                        total += grid[nr as usize * n + nc as usize];
                        count += 1;
                    }
                }
                g![r, c] = total / count as f64 + (rng.unit() * 2.0 - 1.0) * scale;
                c += step;
            }
            r += half;
        }

        scale *= 0.5;
        step   = half;
    }

    grid.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
    grid
}

// --- Public entry point ---------------------------------------------------

/// Fill `layer_buf` (f32 linear-light RGBA, row-major, W×H×4) with plasma.
/// seed and turbulence match spec/plasma.md.
pub fn apply_plasma(layer_buf: &mut [f32], seed: u64, turbulence: f64) {
    let seed_r = seed;
    let seed_g = seed ^ 0x9E3779B97F4A7C15u64;
    let seed_b = seed ^ 0xD1B54A32D192ED03u64;

    let r_grid = diamond_square(&mut Rng::new(seed_r), turbulence);
    let g_grid = diamond_square(&mut Rng::new(seed_g), turbulence);
    let b_grid = diamond_square(&mut Rng::new(seed_b), turbulence);

    for row in 0..H {
        for col in 0..W {
            let src = row * GRID + col;
            let dst = (row * W + col) * 4;
            layer_buf[dst    ] = r_grid[src] as f32;
            layer_buf[dst + 1] = g_grid[src] as f32;
            layer_buf[dst + 2] = b_grid[src] as f32;
            layer_buf[dst + 3] = 1.0; // opaque
        }
    }
}
