import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def band_to_rgb(img, bands=(3, 2, 1), lower_pct=2, upper_pct=98):
    """
    img: (13, H, W) RAW (unnormalized) array.
    bands: (R, G, B) band indices. Default (3, 2, 1) assumes the standard
           Sentinel-2 L1C 13-band order:
           B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12
           -> index 3 = B04 (Red), index 2 = B03 (Green), index 1 = B02 (Blue).
           If your composites look wrong (odd colors, not like aerial photos),
           verify your file's band order and adjust this tuple.
    Returns (H, W, 3) float in [0, 1], ready for imshow.
    """
    rgb = img[list(bands)].astype(np.float32)       # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))               # (H, W, 3)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [lower_pct, upper_pct])
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def plot_prediction_grid(images, true_idx, pred_idx, probs, class_names,
                         out_path, n_cols=4, title=None, bands=(3, 2, 1)):
    """
    images: (N, 13, H, W) raw arrays
    true_idx, pred_idx: (N,) int label indices
    probs: (N,) confidence of the predicted class
    """
    n = len(images)
    n_cols = min(n_cols, max(n, 1))
    n_rows = math.ceil(n / n_cols) if n else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        rgb = band_to_rgb(images[i], bands)
        ax.imshow(rgb)
        correct = true_idx[i] == pred_idx[i]
        color = "green" if correct else "red"
        txt = (f"True: {class_names[true_idx[i]]}\n"
               f"Pred: {class_names[pred_idx[i]]} ({probs[i]*100:.1f}%)")
        ax.set_title(txt, fontsize=9, color=color)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
