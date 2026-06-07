"""QBist genetic abstract-pattern fill — see spec/qbist.md for full contract.

NO linear→sRGB conversion: reg[0] bytes are written directly (unlike plasma).
All register arithmetic in f64.
"""

import argparse
import math
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRANSFORMS  = 36
REGISTERS   = 6
OPCODES     = 9

PROJECTION  = 0
SHIFT       = 1
SHIFTBACK   = 2
ROTATE      = 3
ROTATE2     = 4
MULTIPLY    = 5
COMPLEMENT  = 6
SINE        = 7
CONDITIONAL = 8

MASK64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# PRNG: xorshift64* with splitmix64 seed avalanche
# ---------------------------------------------------------------------------

def splitmix64(seed: int) -> int:
    z = (seed + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


class RNG:
    def __init__(self, seed: int):
        # Avalanche seed before first draw — prevents all-zero draws for seed=0.
        self.state = splitmix64(seed & MASK64)

    def unit(self) -> float:
        s = self.state & MASK64
        s ^= (s >> 12) & MASK64
        s ^= (s << 25) & MASK64
        s ^= (s >> 27) & MASK64
        self.state = s
        r = (s * 0x2545F4914F6CDD1D) & MASK64
        return (r >> 11) / 9007199254740992.0

    def rng_range(self, n: int) -> int:
        return int(self.unit() * n)


# ---------------------------------------------------------------------------
# Genome creation and optimisation
# ---------------------------------------------------------------------------

def create_info(seed: int):
    """Return (transformSequence, source, control, dest) each length 36."""
    rng = RNG(seed)
    ts, src, ctl, dst = [], [], [], []
    for _ in range(TRANSFORMS):
        ts.append(rng.rng_range(OPCODES))
        src.append(rng.rng_range(REGISTERS))
        ctl.append(rng.rng_range(REGISTERS))
        dst.append(rng.rng_range(REGISTERS))
    return ts, src, ctl, dst


def optimize(ts, src, ctl, dst):
    """Return (used_trans_flag[36], used_reg_flag[6]).

    Mutates ctl in-place for ROTATE/ROTATE2/COMPLEMENT (sets ctl[i]=dst[i]).
    """
    # Step 1: fixup degenerate transforms
    for i in range(TRANSFORMS):
        if ts[i] in (ROTATE, ROTATE2, COMPLEMENT):
            ctl[i] = dst[i]

    used_trans = [False] * TRANSFORMS
    used_reg   = [False] * REGISTERS

    def check_last_modified(index: int, reg: int) -> None:
        i = index - 1
        while i >= 0 and dst[i] != reg:
            i -= 1
        if i < 0:
            used_reg[reg] = True
        else:
            used_trans[i] = True
            check_last_modified(i, src[i])
            check_last_modified(i, ctl[i])

    check_last_modified(TRANSFORMS, 0)
    return used_trans, used_reg


# ---------------------------------------------------------------------------
# Renderer (numpy-vectorised over columns, scalar over rows/supersamples)
# ---------------------------------------------------------------------------

def render(genome, used_trans, used_reg, w: int, h: int, os: int) -> np.ndarray:
    """Render qbist to (h, w, 4) uint8 RGBA.

    No sRGB conversion — reg[0] values written directly per spec.
    """
    ts_arr, src_arr, ctl_arr, dst_arr = genome

    used_reg_list   = [r for r in range(REGISTERS)  if used_reg[r]]
    used_trans_list = [t for t in range(TRANSFORMS) if used_trans[t]]

    rgba  = np.zeros((h, w, 4), dtype=np.uint8)
    cols  = np.arange(w, dtype=np.float64)

    for row in range(h):
        accum = np.zeros((w, 3), dtype=np.float64)

        for ys in range(os):
            y_val = (row * os + ys) / (h * os)

            for xs in range(os):
                x_arr = (cols * os + xs) / (w * os)  # shape (w,)

                # reg[r] has shape (w, 3) — list so assignment replaces refs cleanly
                reg = [np.zeros((w, 3), dtype=np.float64) for _ in range(REGISTERS)]

                for i, r in enumerate(used_reg_list):
                    reg[r][:, 0] = x_arr
                    reg[r][:, 1] = y_val
                    reg[r][:, 2] = i / 6.0

                for t in used_trans_list:
                    op = ts_arr[t]
                    s  = src_arr[t]
                    c  = ctl_arr[t]
                    d  = dst_arr[t]
                    sv = reg[s]  # view; reg[d] = new_array never aliases sv
                    cv = reg[c]

                    if op == PROJECTION:
                        scalar = np.einsum('ij,ij->i', sv, cv)[:, np.newaxis] / 3.0
                        reg[d] = scalar * cv
                    elif op == SHIFT:
                        v = sv + cv
                        reg[d] = np.where(v > 1.0, v - 1.0, v)
                    elif op == SHIFTBACK:
                        v = cv - sv
                        reg[d] = np.where(v < 0.0, v + 1.0, v)
                    elif op == ROTATE:
                        reg[d] = sv[:, [1, 2, 0]]
                    elif op == ROTATE2:
                        reg[d] = sv[:, [2, 0, 1]]
                    elif op == MULTIPLY:
                        reg[d] = sv * cv
                    elif op == COMPLEMENT:
                        reg[d] = 1.0 - sv
                    elif op == SINE:
                        reg[d] = (1.0 + np.sin(20.0 * sv)) / 2.0
                    elif op == CONDITIONAL:
                        cond = (cv[:, 0] + cv[:, 1] + cv[:, 2]) > 0.5
                        reg[d] = np.where(cond[:, np.newaxis], sv, cv)

                # Inner quantisation: GIMP two-stage truncation (spec §Rendering)
                accum += np.trunc(reg[0] * 255.0 + 0.5)

        # Outer quantisation
        out = np.floor(accum / (os * os) + 0.5).clip(0, 255).astype(np.uint8)
        rgba[row, :, :3] = out
        rgba[row, :, 3]  = 255

    return rgba


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",         type=lambda x: int(x, 0), default=0)
    ap.add_argument("--oversampling", type=int,   default=4)
    ap.add_argument("--size",         type=int,   default=3500)
    args = ap.parse_args()

    seed = args.seed & MASK64
    genome = create_info(seed)
    ts, src, ctl, dst = genome
    used_trans, used_reg = optimize(ts, src, ctl, dst)

    active_trans = sum(used_trans)
    active_regs  = sum(used_reg)
    print(f"seed={seed}  used_reg_flag={used_reg}  active_transforms={active_trans}  active_regs={active_regs}")

    rgba = render(genome, used_trans, used_reg, args.size, args.size, args.oversampling)
    img  = Image.fromarray(rgba, "RGBA")
    img.save("qbist.png", dpi=(600, 600))
    print(f"Wrote qbist.png  size={args.size}  os={args.oversampling}")


if __name__ == "__main__":
    main()
