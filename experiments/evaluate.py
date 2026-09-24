import argparse, json, os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets.eurosat_dataset import EuroSATMS
from models.classifier import BDHClassifier
from utils.metrics import run_eval, full_metrics, plot_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"results/checkpoints/{args.name}_best.pt", map_location=device)
    model = BDHClassifier(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")

    train_df = pd.read_csv("data/splits/train.csv")
    class_names = train_df.drop_duplicates("label").sort_values("label")["class"].tolist()

    test_ds = EuroSATMS("test", train=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    loss, acc, preds, labels = run_eval(model, loader, device, nn.CrossEntropyLoss())
    metrics = full_metrics(labels, preds)
    print(f"\nTEST  loss {loss:.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n" + classification_report(labels, preds, target_names=class_names, digits=4))

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    with open(f"results/tables/{args.name}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion(labels, preds, class_names, f"results/figures/{args.name}_confusion.png")
    print(f"Saved metrics and confusion matrix for '{args.name}'")


if __name__ == "__main__":
    main()