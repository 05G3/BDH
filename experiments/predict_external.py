import argparse
import json

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.classifier import BDHClassifier
from utils.visualization import band_to_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to a 13-band GeoTIFF")
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--out", default="results/figures/external_prediction.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"results/checkpoints/{args.name}_best.pt", map_location=device)
    model = BDHClassifier(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std = torch.tensor(stats["std"]).view(1, 13, 1, 1)

    train_df = pd.read_csv("data/splits/train.csv")
    class_names = train_df.drop_duplicates("label").sort_values("label")["class"].tolist()

    with rasterio.open(args.image) as src:
        img = src.read().astype(np.float32)    # (13, H, W)
    assert img.shape[0] == 13, f"Expected 13 bands, got {img.shape[0]}"
    if img.shape[1:] != (64, 64):
        print(f"Warning: image is {img.shape[1:]}, model expects 64x64 — results may degrade")

    x = torch.from_numpy(img).unsqueeze(0)
    xn = (x - mean) / (std + 1e-6)
    with torch.no_grad():
        logits = model(xn.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())

    rgb = band_to_rgb(img)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(f"Predicted: {class_names[pred]} ({probs[pred]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print("Top-3 predictions:")
    for i in np.argsort(-probs)[:3]:
        print(f"  {class_names[i]:22s} {probs[i]*100:5.2f}%")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
