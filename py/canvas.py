from PIL import Image

W = H = 3500
PPI = 600

img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
img.save("canvas.png", dpi=(PPI, PPI))
print(f"Wrote canvas.png  {W}x{H} RGBA  {PPI} PPI")
