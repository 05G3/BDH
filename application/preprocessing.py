import io
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def validate_geotiff_bands(arr: np.ndarray) -> np.ndarray:
    """Ensures input array has at least 13 spectral bands for Sentinel-2."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C, H, W), got {arr.shape}.")
    if arr.shape[0] < 13:
        raise ValueError(
            f"GeoTIFF has only {arr.shape[0]} bands. BDH requires all 13 Sentinel-2 bands "
            "(B01-B12, including B8A)."
        )
    return arr[:13]


def load_geotiff(file_or_bytes: Union[str, bytes, io.BytesIO]) -> np.ndarray:
    """Reads a Sentinel-2 GeoTIFF from path or in-memory bytes."""
    import rasterio
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)
    with rasterio.open(file_or_bytes) as src:
        arr = src.read().astype(np.float32)
    return validate_geotiff_bands(arr)


def load_rgb_image(file_or_bytes: Union[str, bytes, io.BytesIO], mean_vals: List[float]) -> np.ndarray:
    """
    Reads an RGB image (PNG, JPG) and prepares a synthetic 13-band Sentinel-2 representation:
    Maps Red -> Band 4, Green -> Band 3, Blue -> Band 2.
    Estimates Near-Infrared B08 based on green/red reflectance.
    Fills other bands with training mean statistics.
    """
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)
    img_pil = Image.open(file_or_bytes).convert("RGB")
    W, H = img_pil.size
    # If smaller than 64x64, resize up
    if H < 64 or W < 64:
        img_pil = img_pil.resize((max(64, W), max(64, H)), Image.Resampling.BILINEAR)
        W, H = img_pil.size

    rgb = np.array(img_pil, dtype=np.float32)  # (H, W, 3), range [0..255]
    arr = np.zeros((13, H, W), dtype=np.float32)

    for b in range(13):
        arr[b, :, :] = mean_vals[b]

    # Approximate reflectance scale from dataset means
    scale = (mean_vals[3] + mean_vals[2] + mean_vals[1]) / (3.0 * 128.0)
    arr[3] = rgb[:, :, 0] * scale  # B04 (Red)
    arr[2] = rgb[:, :, 1] * scale  # B03 (Green)
    arr[1] = rgb[:, :, 2] * scale  # B02 (Blue)
    arr[7] = np.maximum(arr[3], arr[2]) * 1.3  # B08 (NIR proxy)
    return arr


def extract_sliding_tiles(
    arr: np.ndarray, tile_size: int = 64, stride: int = 64
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Splits an arbitrary-sized (13, H, W) satellite scene into 64x64 tiles for BDH.
    Returns:
        tiles: list of (13, 64, 64) numpy arrays
        coords: list of (y1, x1, y2, x2) bounding boxes
        (H, W): original spatial dimensions
    """
    _, H, W = arr.shape
    if H == tile_size and W == tile_size:
        return [arr], [(0, 0, tile_size, tile_size)], (H, W)

    tiles = []
    coords = []

    # Calculate steps
    y_starts = list(range(0, max(1, H - tile_size + 1), stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)

    x_starts = list(range(0, max(1, W - tile_size + 1), stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    for y in y_starts:
        for x in x_starts:
            tile = arr[:, y : y + tile_size, x : x + tile_size]
            # Ensure tile is exact tile_size
            if tile.shape[1:] == (tile_size, tile_size):
                tiles.append(tile)
                coords.append((y, x, y + tile_size, x + tile_size))

    return tiles, coords, (H, W)


def normalize_batch(tiles: List[np.ndarray], mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    """Converts a list of (13, 64, 64) numpy tiles into a normalized (B, 13, 64, 64) PyTorch tensor."""
    batch_np = np.stack(tiles, axis=0)  # (B, 13, 64, 64)
    x = torch.from_numpy(batch_np).float()
    return (x - mean_t) / (std_t + 1e-6)
