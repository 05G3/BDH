import argparse
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.classifier import BDHClassifier
from utils.visualization import plot_prediction_grid


def load_model(name, device):
    ckpt = torch.load(f"results/checkpoints/{name}_best.pt", map_location=device)
    model = BDHClassifier(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--n_correct", type=int, default=16)
    p.add_argument("--n_incorrect", type=int, default=16)
    p.add_argument("--bands", type=int, nargs=3, default=[3, 2, 1],
                   help="R G B band indices (default: Sentinel-2 B04,B03,B02)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.name, device)
    print(f"Loaded {args.name} (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})")

    images = np.load("data/processed/test_images.npy")   # raw, unnormalized (N,13,64,64)
    labels = np.load("data/processed/test_labels.npy")
    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std = torch.tensor(stats["std"]).view(1, 13, 1, 1)

    train_df = pd.read_csv("data/splits/train.csv")
    class_names = train_df.drop_duplicates("label").sort_values("label")["class"].tolist()

    all_preds, all_probs = [], []
    bs = 128
    with torch.no_grad():
        for i in range(0, len(images), bs):
            x = torch.from_numpy(images[i:i + bs]).float()
            x = (x - mean) / (std + 1e-6)
            logits = model(x.to(device))
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(conf.cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    correct_mask = preds == labels
    print(f"Test accuracy (recomputed here): {correct_mask.mean():.4f}")

    rng = np.random.default_rng(args.seed)
    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(~correct_mask)[0]
    print(f"Correct: {len(correct_idx)}  Incorrect: {len(incorrect_idx)}")

    def sample(idx_pool, n):
        idx_pool = idx_pool.copy()
        rng.shuffle(idx_pool)
        return idx_pool[:n]

    os.makedirs("results/figures", exist_ok=True)

    # 1) Grid of correctly classified examples
    sel = sample(correct_idx, args.n_correct)
    plot_prediction_grid(images[sel], labels[sel], preds[sel], probs[sel], class_names,
                         f"results/figures/{args.name}_correct_samples.png",
                         title=f"{args.name}: correctly classified test samples",
                         bands=tuple(args.bands))
    print(f"Saved results/figures/{args.name}_correct_samples.png")

    # 2) Grid of misclassified examples (if any)
    if len(incorrect_idx) > 0:
        sel = sample(incorrect_idx, args.n_incorrect)
        plot_prediction_grid(images[sel], labels[sel], preds[sel], probs[sel], class_names,
                             f"results/figures/{args.name}_misclassified_samples.png",
                             title=f"{args.name}: misclassified test samples",
                             bands=tuple(args.bands))
        print(f"Saved results/figures/{args.name}_misclassified_samples.png")
    else:
        print("No misclassified samples found — nothing to plot.")

    # 3) One correctly classified example per class — good paper figure
    per_class = []
    for c in range(len(class_names)):
        cand = correct_idx[labels[correct_idx] == c]
        if len(cand) > 0:
            per_class.append(cand[0])
    if per_class:
        per_class = np.array(per_class)
        plot_prediction_grid(images[per_class], labels[per_class], preds[per_class],
                             probs[per_class], class_names,
                             f"results/figures/{args.name}_per_class_examples.png",
                             title=f"{args.name}: one example per class", n_cols=5,
                             bands=tuple(args.bands))
        print(f"Saved results/figures/{args.name}_per_class_examples.png")


if __name__ == "__main__":
    main()
