# basecoat

Generate 2D PNG/JPG stills from algorithms.

## Workflow

```
spec/   ← language-agnostic technique specs (Markdown) — the contract
py/     ← Python impl (Pillow + NumPy + scikit-image) — exploration
rs/     ← Rust impl (image 0.25 + png 0.17) — production output
tools/  ← pixdiff.py comparator
```

**Python = explore.** Iterate quickly in Python to get the algorithm right.
**Rust = keepers.** Once the Python version passes the diff, port to Rust for
the final high-quality output.
**spec = contract.** Every technique is fully specified in `spec/` before
either implementation is written. Dimensions, colour model, resolution
metadata, and numeric conventions are pinned there, not inferred from code.

## Running the pilot technique (transparent canvas)

```bash
# Python
cd py
pip install -r requirements.txt
python canvas.py          # → py/canvas.png

# Rust
cd rs
cargo run --bin canvas    # → rs/canvas.png

# Diff (from repo root, tolerance=0 for lossless PNG)
python tools/pixdiff.py py/canvas.png rs/canvas.png
```

Expected:

```
size        : 3500x3500
max delta   : 0
diff pixels : 0
tolerance   : 0
RESULT: PASS
```

## Adding a new technique

1. Write `spec/<name>.md` — pin all numeric and convention choices.
2. Implement `py/<name>.py`; iterate until the image looks right.
3. Implement `rs/src/bin/<name>.rs`.
4. Add a `pixdiff` check to confirm the two outputs agree.

## Tools

### `tools/pixdiff.py`

```
python tools/pixdiff.py <image-a> <image-b> [--tolerance N]
```

Compares two images channel-by-channel. Default tolerance is 0 (exact match,
suitable for PNG). Use `--tolerance 2` or higher when comparing lossy formats
(JPEG) or when floating-point rounding differs between implementations.

Exits 0 on pass, 1 on fail.
