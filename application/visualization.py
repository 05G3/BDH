from typing import List, Tuple

import numpy as np
from PIL import Image

from utils.visualization import band_to_rgb

# Standard cartographic colors for the 10 EuroSAT land-cover classes
CLASS_PALETTE = {
    0: (245, 158, 11),   # AnnualCrop: Warm Gold
    1: (4, 120, 87),     # Forest: Deep Green
    2: (16, 185, 129),   # HerbaceousVegetation: Bright Green
    3: (75, 85, 99),     # Highway: Dark Slate / Asphalt
    4: (185, 28, 28),    # Industrial: Dark Red
    5: (132, 204, 22),   # Pasture: Lime Green
    6: (217, 119, 6),    # PermanentCrop: Amber / Ochre
    7: (236, 72, 153),   # Residential: Magenta / Rose
    8: (6, 182, 212),    # River: Cyan / Azure
    9: (29, 78, 216),    # SeaLake: Deep Royal Blue
}


def get_rgb_preview(arr: np.ndarray) -> np.ndarray:
    """Returns (H, W, 3) float in [0, 1] using B04 (Red), B03 (Green), B02 (Blue)."""
    return band_to_rgb(arr, bands=(3, 2, 1))


def get_nir_preview(arr: np.ndarray) -> np.ndarray:
    """Returns (H, W, 3) float in [0, 1] using B08 (NIR), B04 (Red), B03 (Green)."""
    return band_to_rgb(arr, bands=(7, 3, 2))


def render_landcover_map(seg_map: np.ndarray) -> Image.Image:
    """Renders a 2D integer class map into a colorized RGB land-cover map."""
    H, W = seg_map.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for c_id, color in CLASS_PALETTE.items():
        mask = seg_map == c_id
        colored[mask] = color
    return Image.fromarray(colored)


def create_segmented_overlay(
    rgb_composite: np.ndarray,
    seg_map: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blends the colorized land-cover segmentation map on top of the true-color satellite imagery."""
    H, W, _ = rgb_composite.shape
    base_rgb = (np.clip(rgb_composite, 0, 1) * 255).astype(np.uint8)
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    for c_id, color in CLASS_PALETTE.items():
        mask = seg_map == c_id
        color_map[mask] = color

    # Blend where valid prediction exists
    valid_mask = seg_map >= 0
    blended = base_rgb.copy()
    blended[valid_mask] = (
        (1.0 - alpha) * base_rgb[valid_mask] + alpha * color_map[valid_mask]
    ).astype(np.uint8)

    return Image.fromarray(blended)
