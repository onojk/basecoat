"""Diamond-square plasma — see spec/plasma.md for full algorithm contract."""

import argparse
import numpy as np
from PIL import Image

W = H = 3500
PPI = 600
N = 4097  # 2^12 + 1


# --- PRNG: xorshift64* ---------------------------------------------------

def xorshift64star(state: int) -> tuple[int, int]:
    state &= 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
    result = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    return state, result


class RNG:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def unit(self) -> float:
        self.state, r = xorshift64star(self.state)
        return (r >> 11) / 9007199254740992.0  # top 53 bits / 2^53


# --- Diamond-square -------------------------------------------------------

def diamond_square(rng: RNG, turbulence: float) -> np.ndarray:
    grid = np.zeros((N, N), dtype=np.float64)

    # Seed corners row-major: (0,0), (0,N-1), (N-1,0), (N-1,N-1)
    grid[0,   0  ] = rng.unit()
    grid[0,   N-1] = rng.unit()
    grid[N-1, 0  ] = rng.unit()
    grid[N-1, N-1] = rng.unit()

    scale = turbulence
    step = N - 1  # 4096

    while step > 1:
        half = step // 2

        # Diamond step — row-major over square centers
        for r in range(0, N - 1, step):
            for c in range(0, N - 1, step):
                mean = (grid[r, c] + grid[r, c + step] +
                        grid[r + step, c] + grid[r + step, c + step]) * 0.25
                grid[r + half, c + half] = mean + (rng.unit() * 2.0 - 1.0) * scale

        # Square step — row-major over diamond edge midpoints.
        # Even row-bands (r/half even) start at column=half;
        # odd row-bands start at column=0.  Both step by `step`.
        for r in range(0, N, half):
            start = half if (r // half) % 2 == 0 else 0
            for c in range(start, N, step):
                total, count = 0.0, 0
                for dr, dc in [(-half, 0), (half, 0), (0, -half), (0, half)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        total += grid[nr, nc]
                        count += 1
                grid[r, c] = total / count + (rng.unit() * 2.0 - 1.0) * scale

        scale *= 0.5
        step = half

    return np.clip(grid, 0.0, 1.0)


# --- sRGB encode ----------------------------------------------------------

def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(
        v <= 0.0031308,
        v * 12.92,
        1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055,
    )


# --- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0)
    ap.add_argument("--turbulence", type=float, default=1.0)
    args = ap.parse_args()

    seed = args.seed & 0xFFFFFFFFFFFFFFFF
    SEED_R = seed
    SEED_G = (seed ^ 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    SEED_B = (seed ^ 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF

    r_lin = diamond_square(RNG(SEED_R), args.turbulence)[:H, :W]
    g_lin = diamond_square(RNG(SEED_G), args.turbulence)[:H, :W]
    b_lin = diamond_square(RNG(SEED_B), args.turbulence)[:H, :W]

    r8 = (linear_to_srgb(r_lin) * 255.0 + 0.5).astype(np.uint8)
    g8 = (linear_to_srgb(g_lin) * 255.0 + 0.5).astype(np.uint8)
    b8 = (linear_to_srgb(b_lin) * 255.0 + 0.5).astype(np.uint8)
    a8 = np.full((H, W), 255, dtype=np.uint8)

    rgba = np.stack([r8, g8, b8, a8], axis=-1)
    img = Image.fromarray(rgba, "RGBA")
    img.save("plasma.png", dpi=(PPI, PPI))
    print(f"Wrote plasma.png  seed={args.seed}  turbulence={args.turbulence}")


if __name__ == "__main__":
    main()
