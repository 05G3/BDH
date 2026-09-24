import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return total_loss / n, accuracy_score(labels, preds), preds, labels


def full_metrics(labels, preds):
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, preds)),
            "precision_macro": float(p),
            "recall_macro": float(r),
            "f1_macro": float(f1)}


def plot_confusion(labels, preds, class_names, out_path, normalize=True):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)