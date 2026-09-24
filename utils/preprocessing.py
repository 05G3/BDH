import os, glob, json
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split

def build_index(root: str) -> pd.DataFrame:
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rows = []
    for c in classes:
        for p in sorted(glob.glob(os.path.join(root, c, "*.tif"))):
            rows.append({"path": p, "label": class_to_idx[c], "class": c})
    return pd.DataFrame(rows), class_to_idx

def make_splits(df, out_dir="data/splits", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed)
    # split temp 50/50 -> 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)

    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    return train_df, val_df, test_df

def read_image(path: str) -> np.ndarray:
    """Returns float32 array of shape (13, 64, 64)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def compute_train_stats(train_df, out_path="data/splits/norm_stats.json"):
    """Per-band mean/std from the TRAINING set only (streaming, memory-safe)."""
    n_pixels = 0
    s = np.zeros(13, dtype=np.float64)
    s2 = np.zeros(13, dtype=np.float64)
    for p in train_df["path"]:
        x = read_image(p).reshape(13, -1).astype(np.float64)
        s += x.sum(1)
        s2 += (x ** 2).sum(1)
        n_pixels += x.shape[1]
    mean = s / n_pixels
    std = np.sqrt(s2 / n_pixels - mean ** 2)
    with open(out_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    return mean, std