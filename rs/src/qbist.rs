//! QBist genetic abstract-pattern fill — see spec/qbist.md.
//! All register arithmetic in f64. No sRGB conversion (reg[0] bytes written directly).

use std::sync::atomic::{AtomicU32, Ordering};

pub const TRANSFORMS: usize = 36;
pub const REGISTERS:  usize = 6;
pub const OPCODES:    usize = 9;

const PROJECTION:  u8 = 0;
const SHIFT:       u8 = 1;
const SHIFTBACK:   u8 = 2;
const ROTATE:      u8 = 3;
const ROTATE2:     u8 = 4;
const MULTIPLY:    u8 = 5;
const COMPLEMENT:  u8 = 6;
const SINE:        u8 = 7;
const CONDITIONAL: u8 = 8;

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

/// One-shot splitmix64 avalanche — maps seed to a non-zero xorshift64* state.
/// Without this, seed=0 produces all-zero draws (xorshift64*(0) = 0 forever).
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: splitmix64(seed) }
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

    pub fn range(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize
    }
}

// ---------------------------------------------------------------------------
// Genome
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Genome {
    pub ts:  [u8;  TRANSFORMS],
    pub src: [u8;  TRANSFORMS],
    pub ctl: [u8;  TRANSFORMS],
    pub dst: [u8;  TRANSFORMS],
}

/// Build genome from seed — 4 draws per k in order: ts, src, ctl, dst.
pub fn create_info(seed: u64) -> Genome {
    let mut rng = Rng::new(seed);
    let mut g   = Genome { ts: [0; TRANSFORMS], src: [0; TRANSFORMS],
                           ctl: [0; TRANSFORMS], dst: [0; TRANSFORMS] };
    for k in 0..TRANSFORMS {
        g.ts[k]  = rng.range(OPCODES)   as u8;
        g.src[k] = rng.range(REGISTERS) as u8;
        g.ctl[k] = rng.range(REGISTERS) as u8;
        g.dst[k] = rng.range(REGISTERS) as u8;
    }
    g
}

// ---------------------------------------------------------------------------
// Optimise
// ---------------------------------------------------------------------------

/// Backward-dependency walk; returns (used_trans[36], used_reg[6]).
/// Mutates g.ctl for ROTATE/ROTATE2/COMPLEMENT.
pub fn optimize(g: &mut Genome) -> ([bool; TRANSFORMS], [bool; REGISTERS]) {
    // Step 1: fixup — these opcodes ignore ctl; redirect to avoid spurious deps
    for i in 0..TRANSFORMS {
        if matches!(g.ts[i], ROTATE | ROTATE2 | COMPLEMENT) {
            g.ctl[i] = g.dst[i];
        }
    }

    let mut used_trans = [false; TRANSFORMS];
    let mut used_reg   = [false; REGISTERS];

    check_last_modified(g, TRANSFORMS, 0, &mut used_trans, &mut used_reg);
    (used_trans, used_reg)
}

fn check_last_modified(
    g:          &Genome,
    index:      usize,
    reg:        u8,
    used_trans: &mut [bool; TRANSFORMS],
    used_reg:   &mut [bool; REGISTERS],
) {
    let mut i = index as isize - 1;
    while i >= 0 && g.dst[i as usize] != reg {
        i -= 1;
    }
    if i < 0 {
        used_reg[reg as usize] = true;
    } else {
        let ui = i as usize;
        used_trans[ui] = true;
        check_last_modified(g, ui, g.src[ui], used_trans, used_reg);
        check_last_modified(g, ui, g.ctl[ui], used_trans, used_reg);
    }
}

// ---------------------------------------------------------------------------
// Transform application (scalar, f64)
// ---------------------------------------------------------------------------

#[inline(always)]
fn apply(opcode: u8, sv: [f64; 3], cv: [f64; 3]) -> [f64; 3] {
    match opcode {
        PROJECTION => {
            let dot = sv[0]*cv[0] + sv[1]*cv[1] + sv[2]*cv[2];
            let sc  = dot / 3.0;
            [sc*cv[0], sc*cv[1], sc*cv[2]]
        }
        SHIFT => [
            { let v = sv[0]+cv[0]; if v > 1.0 { v-1.0 } else { v } },
            { let v = sv[1]+cv[1]; if v > 1.0 { v-1.0 } else { v } },
            { let v = sv[2]+cv[2]; if v > 1.0 { v-1.0 } else { v } },
        ],
        SHIFTBACK => [
            { let v = cv[0]-sv[0]; if v < 0.0 { v+1.0 } else { v } },
            { let v = cv[1]-sv[1]; if v < 0.0 { v+1.0 } else { v } },
            { let v = cv[2]-sv[2]; if v < 0.0 { v+1.0 } else { v } },
        ],
        ROTATE  => [sv[1], sv[2], sv[0]],
        ROTATE2 => [sv[2], sv[0], sv[1]],
        MULTIPLY => [sv[0]*cv[0], sv[1]*cv[1], sv[2]*cv[2]],
        COMPLEMENT => [1.0-sv[0], 1.0-sv[1], 1.0-sv[2]],
        SINE => [
            (1.0 + (20.0*sv[0]).sin()) / 2.0,
            (1.0 + (20.0*sv[1]).sin()) / 2.0,
            (1.0 + (20.0*sv[2]).sin()) / 2.0,
        ],
        CONDITIONAL => {
            if cv[0]+cv[1]+cv[2] > 0.5 { sv } else { cv }
        }
        _ => sv,
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Render qbist to RGBA u8 bytes (row-major, w*h*4), reporting progress.
///
/// `progress` is incremented to the completed row count after each row finishes
/// (range 0..=h). Callers can read it with `Ordering::Relaxed` for display.
/// Math is identical to `render`; the delta-0 contract still holds.
pub fn render_with_progress(
    g:          &Genome,
    used_trans: &[bool; TRANSFORMS],
    used_reg:   &[bool; REGISTERS],
    w:          usize,
    h:          usize,
    os:         usize,
    progress:   &AtomicU32,
) -> Vec<u8> {
    let reg_list:   Vec<usize> = (0..REGISTERS).filter(|&r| used_reg[r]).collect();
    let trans_list: Vec<usize> = (0..TRANSFORMS).filter(|&t| used_trans[t]).collect();

    let os_sq = (os * os) as f64;
    let mut out = vec![255u8; w * h * 4];

    for row in 0..h {
        for col in 0..w {
            let mut accum = [0.0f64; 3];

            for ys in 0..os {
                for xs in 0..os {
                    let x_norm = (col * os + xs) as f64 / (w * os) as f64;
                    let y_norm = (row * os + ys) as f64 / (h * os) as f64;

                    let mut reg = [[0.0f64; 3]; REGISTERS];

                    for (i, &r) in reg_list.iter().enumerate() {
                        reg[r] = [x_norm, y_norm, i as f64 / 6.0];
                    }

                    for &t in &trans_list {
                        let sv = reg[g.src[t] as usize];
                        let cv = reg[g.ctl[t] as usize];
                        reg[g.dst[t] as usize] = apply(g.ts[t], sv, cv);
                    }

                    // Inner quantisation: trunc(reg[0][c]*255+0.5)
                    for c in 0..3 {
                        accum[c] += (reg[0][c] * 255.0 + 0.5).trunc();
                    }
                }
            }

            // Outer quantisation: floor(accum/(os²)+0.5) clamped to u8
            let base = (row * w + col) * 4;
            for c in 0..3 {
                let v = (accum[c] / os_sq + 0.5).floor().clamp(0.0, 255.0) as u8;
                out[base + c] = v;
            }
            // out[base+3] = 255 already (vec initialised to 255)
        }
        // Report row completion — ~once per row, not per pixel.
        progress.store(row as u32 + 1, Ordering::Relaxed);
    }

    out
}

/// Render qbist to RGBA u8 bytes (row-major, w*h*4).
/// No sRGB conversion — reg[0] values written directly per spec.
/// Delegates to `render_with_progress` with a throwaway atomic.
pub fn render(
    g:          &Genome,
    used_trans: &[bool; TRANSFORMS],
    used_reg:   &[bool; REGISTERS],
    w:          usize,
    h:          usize,
    os:         usize,
) -> Vec<u8> {
    let dummy = AtomicU32::new(0);
    render_with_progress(g, used_trans, used_reg, w, h, os, &dummy)
}
