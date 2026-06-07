"""Reference PNGs for qbist pixdiff validation — see spec/qbist.md.

Reference PNGs written (names match rs/src/bin/test_qbist.rs):
  test_qbist_seed0_os1.png    (256, seed 0,  os 1)
  test_qbist_seed0_os4.png    (256, seed 0,  os 4)
  test_qbist_seed1_os4.png    (256, seed 1,  os 4)
  test_qbist_seed42_os4.png   (256, seed 42, os 4)
  test_qbist_confirm3500.png  (3500, seed 0, os 4)
"""
from __future__ import annotations
from PIL import Image
from qbist import create_info, optimize, render, MASK64

PPI = (600.0, 600.0)

CASES = [
    # (name,          size, seed, os)
    ("seed0_os1",     256,  0,    1),
    ("seed0_os4",     256,  0,    4),
    ("seed1_os4",     256,  1,    4),
    ("seed42_os4",    256,  42,   4),
    ("confirm3500",   3500, 0,    4),
]


def run_case(name: str, size: int, seed: int, os: int) -> None:
    seed = seed & MASK64
    genome = create_info(seed)
    ts, src, ctl, dst = genome
    used_trans, used_reg = optimize(ts, src, ctl, dst)

    active_t = sum(used_trans)
    active_r = sum(used_reg)
    print(f"  seed={seed} os={os} size={size}  active_transforms={active_t}  active_regs={active_r}  used_reg={used_reg}")

    rgba = render(genome, used_trans, used_reg, size, size, os)
    path = f"test_qbist_{name}.png"
    Image.fromarray(rgba, "RGBA").save(path, dpi=PPI)
    print(f"  wrote {path}")


def main() -> None:
    for name, size, seed, os in CASES:
        print(f"--- {name} ---")
        run_case(name, size, seed, os)
    print("Done.")


if __name__ == "__main__":
    main()
