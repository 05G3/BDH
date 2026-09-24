import glob, os
import numpy as np
import rasterio

ROOT = "data/raw/EuroSAT_MS"

classes = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
print("Classes:", classes, len(classes))

total = 0
for c in classes:
    n = len(glob.glob(os.path.join(ROOT, c, "*.tif")))
    total += n
    print(f"{c:22s} {n}")
print("Total images:", total)   # expect ~27,000

# Look at one image
path = glob.glob(os.path.join(ROOT, classes[0], "*.tif"))[0]
with rasterio.open(path) as src:
    img = src.read()            # (bands, H, W)
    print("File:", path)
    print("Shape:", img.shape)  # expect (13, 64, 64)
    print("dtype:", img.dtype)  # usually uint16
    print("min/max:", img.min(), img.max())
    print("per-band mean:", img.reshape(13, -1).mean(1).round(1))