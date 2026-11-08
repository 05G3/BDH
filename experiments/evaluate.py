import argparse, json, os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets.eurosat_dataset import EuroSATMS
from models.factory import build_model
from utils.metrics import run_eval, full_metrics, plot_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bdh_main")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"results/checkpoints/{args.name}_best.pt", map_location=device)
    model_type = ckpt.get("model_type", "bdh")          # old checkpoints are BDH
    model = build_model(model_type, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.name} [{model_type}] epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")

    train_df = pd.read_csv("data/splits/train.csv")
    class_names = train_df.drop_duplicates("label").sort_values("label")["class"].tolist()

    test_ds = EuroSATMS("test", train=False, bands=ckpt.get("bands"))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    loss, acc, preds, labels = run_eval(model, loader, device, nn.CrossEntropyLoss())
    metrics = full_metrics(labels, preds)
    metrics.update(n_params=n_params, best_epoch=int(ckpt["epoch"]),
                   val_acc=float(ckpt["val_acc"]), model_type=model_type)
    print(f"\nTEST loss {loss:.4f}")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"  {k}: {metrics[k]:.4f}")

    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4, output_dict=True)
    print("\n" + classification_report(labels, preds, target_names=class_names, digits=4))

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    with open(f"results/tables/{args.name}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(report).T.to_csv(f"results/tables/{args.name}_per_class.csv")
    plot_confusion(labels, preds, class_names, f"results/figures/{args.name}_confusion.png")
    print(f"Saved metrics, per-class table and confusion matrix for '{args.name}'")


if __name__ == "__main__":
    main()
