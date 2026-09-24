import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.preprocessing import read_image

os.makedirs("data/processed", exist_ok=True)

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/{split}.csv")
    imgs = np.zeros((len(df), 13, 64, 64), dtype=np.float32)
    labels = df["label"].to_numpy()

    for i, path in enumerate(tqdm(df["path"], desc=f"Caching {split}")):
        imgs[i] = read_image(path)

    np.save(f"data/processed/{split}_images.npy", imgs)
    np.save(f"data/processed/{split}_labels.npy", labels)
    print(f"{split}: {imgs.shape}  {imgs.nbytes / 1e9:.2f} GB  labels {labels.shape}")

print("Done. Cached arrays are in data/processed/")