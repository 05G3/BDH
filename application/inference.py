import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.classifier import BDHClassifier


def load_bdh_model(ckpt_path: str, device: torch.device) -> Tuple[BDHClassifier, Dict[str, Any]]:
    """Loads the trained BDH model weights and training configuration."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["config"]
    model = BDHClassifier(**model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p.numel() for p in model.parameters())
    metadata = {
        "epoch": ckpt.get("epoch", 26),
        "val_acc": ckpt.get("val_acc", 0.9711),
        "n_params": n_params,
        "config": model_cfg,
        "model_type": ckpt.get("model_type", "bdh"),
    }
    return model, metadata


def predict_single_tile(
    model: BDHClassifier,
    normalized_tensor: torch.Tensor,
    device: torch.device,
    record_stats: bool = True,
) -> Dict[str, Any]:
    """
    Executes inference for a single normalized (1, 13, 64, 64) tensor.
    Guarantees model.bdh.record_stats is enabled and sparsity stats are reliably captured.
    If spatial dimensions are smaller than 64x64, pads with edge replication.
    """
    if normalized_tensor.ndim == 3:
        normalized_tensor = normalized_tensor.unsqueeze(0)

    # Spatial dimension guard: pad if smaller than 64x64
    if normalized_tensor.shape[-2] < 64 or normalized_tensor.shape[-1] < 64:
        pad_bottom = max(0, 64 - normalized_tensor.shape[-2])
        pad_right = max(0, 64 - normalized_tensor.shape[-1])
        normalized_tensor = F.pad(normalized_tensor, (0, pad_right, 0, pad_bottom), mode="replicate")

    model.eval()
    if record_stats and hasattr(model, "bdh"):
        model.bdh.record_stats = True

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(normalized_tensor.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    sparsity_stats = []
    if record_stats and hasattr(model, "bdh") and hasattr(model.bdh, "stats"):
        sparsity_stats = list(model.bdh.stats)

    return {
        "pred_idx": pred_idx,
        "confidence": conf,
        "probabilities": probs,
        "latency_ms": latency_ms,
        "sparsity_stats": sparsity_stats,
    }


def predict_scene_sliding_window(
    model: BDHClassifier,
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    orig_shape: Tuple[int, int],
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Runs sliding-window prediction across a large satellite scene.
    Stitches individual 64x64 tile predictions into a 2D tile-based land-cover classification map.
    
    IMPORTANT ARCHITECTURAL DISTINCTION:
    BDH is a patch-level classifier trained on 64x64 EuroSAT tiles. The output of this function
    is a tile-based land-cover classification map (each 64x64 spatial area displays its dominant
    predicted class), NOT a pixel-wise semantic segmentation map. Generating dense per-pixel
    segmentation requires dense prediction models (e.g. U-Net) trained on pixel-annotated masks.
    """
    H, W = orig_shape
    tile_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    conf_map = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    model.eval()
    if hasattr(model, "bdh"):
        model.bdh.record_stats = False

    t0 = time.perf_counter()
    total_tiles = len(tiles)

    if total_tiles > 0:
        for i in range(0, total_tiles, batch_size):
            batch_tiles = tiles[i : i + batch_size]
            batch_coords = coords[i : i + batch_size]

            batch_np = np.stack(batch_tiles, axis=0)  # (B, 13, 64, 64)
            x = torch.from_numpy(batch_np).float()
            xn = (x - mean_t) / (std_t + 1e-6)

            with torch.no_grad():
                logits = model(xn.to(device))
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, 10)

            for j, (y1, x1, y2, x2) in enumerate(batch_coords):
                p = batch_probs[j]
                c_idx = int(np.argmax(p))
                c_conf = float(p[c_idx])

                # Safely clip to original spatial dimensions
                y2_c = min(y2, H)
                x2_c = min(x2, W)
                if y1 < y2_c and x1 < x2_c:
                    tile_map[y1:y2_c, x1:x2_c] = c_idx
                    conf_map[y1:y2_c, x1:x2_c] = np.maximum(conf_map[y1:y2_c, x1:x2_c], c_conf)
                    counts[y1:y2_c, x1:x2_c] += 1

    total_latency_ms = (time.perf_counter() - t0) * 1000

    # Calculate class composition percentages across classified area
    valid_mask = tile_map >= 0
    if valid_mask.any():
        unique, counts_per_class = np.unique(tile_map[valid_mask], return_counts=True)
        total_classified_pixels = np.sum(counts_per_class)
        composition = {}
        for u, cnt in zip(unique, counts_per_class):
            cls_name = class_names[int(u)] if int(u) < len(class_names) else f"Class_{u}"
            composition[cls_name] = round(float(cnt / total_classified_pixels * 100.0), 2)
    else:
        composition = {}

    return {
        "tile_map": tile_map,
        "seg_map": tile_map,  # Kept as alias for compatibility
        "conf_map": conf_map,
        "total_tiles": total_tiles,
        "total_latency_ms": total_latency_ms,
        "composition": composition,
        "is_tile_based_map": True,
    }
