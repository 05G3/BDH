import io
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Canonical 13-band Sentinel-2 order for EuroSAT-MS:
# Index 0: B01 (Coastal aerosol)
# Index 1: B02 (Blue)
# Index 2: B03 (Green)
# Index 3: B04 (Red)
# Index 4: B05 (Vegetation red edge 1)
# Index 5: B06 (Vegetation red edge 2)
# Index 6: B07 (Vegetation red edge 3)
# Index 7: B08 (NIR)
# Index 8: B8A (Narrow NIR)
# Index 9: B09 (Water vapour)
# Index 10: B10 (Cirrus)
# Index 11: B11 (SWIR 1)
# Index 12: B12 (SWIR 2)
CANONICAL_SENTINEL2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
]


def _normalize_band_name(name: str) -> str:
    """Standardizes band identifiers, e.g. 'B1' -> 'B01', 'B8A' -> 'B8A'."""
    clean = name.strip().upper()
    if clean.startswith("BAND_"):
        clean = "B" + clean[5:]
    elif clean.startswith("BAND"):
        clean = "B" + clean[4:]
    if clean.startswith("B") and len(clean) == 2 and clean[1].isdigit():
        clean = f"B0{clean[1]}"
    if clean in ("B08A", "B8A", "BAND8A"):
        clean = "B8A"
    return clean


def validate_geotiff_bands(
    arr: np.ndarray,
    band_names: Optional[List[Optional[str]]] = None,
    nodatavals: Optional[Tuple[Optional[float], ...]] = None,
) -> np.ndarray:
    """
    Validates a multi-spectral Sentinel-2 array for BDH model compatibility:
    1. Verifies 3D structure (C, H, W) and minimum 13 spectral bands.
    2. Validates and handles NaN, Inf, and NoData pixels.
    3. Validates band order against canonical EuroSAT-MS order:
       [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12]
       (reorders if band descriptions are present and permuted).
    4. Validates reflectance scale against EuroSAT training distribution (converts [0, 1] reflectance to DN ~10,000 scale).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D satellite array with shape (C, H, W), got {arr.shape}.")

    n_bands, H, W = arr.shape
    if n_bands < 13:
        raise ValueError(
            f"GeoTIFF contains only {n_bands} bands. BDH requires all 13 Sentinel-2 spectral bands "
            f"in EuroSAT order ({', '.join(CANONICAL_SENTINEL2_BANDS)})."
        )

    # 1. Check for NaN and Inf values
    if np.isnan(arr).any() or np.isinf(arr).any():
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        if nan_mask.all() or inf_mask.all():
            raise ValueError("GeoTIFF array contains only NaN or Infinite values.")
        warnings.warn("GeoTIFF contains NaN/Inf values; replacing with median valid reflectance.")
        for b in range(n_bands):
            valid = arr[b][~(nan_mask[b] | inf_mask[b])]
            med = float(np.median(valid)) if len(valid) > 0 else 0.0
            arr[b][nan_mask[b] | inf_mask[b]] = med

    # 2. Check for nodata values
    if nodatavals is not None:
        for b, nd_val in enumerate(nodatavals[:13]):
            if nd_val is not None and not np.isnan(nd_val):
                nd_mask = (arr[b] == nd_val)
                if nd_mask.all():
                    raise ValueError(f"Band index {b} contains only NoData values ({nd_val}).")
                if nd_mask.any():
                    valid = arr[b][~nd_mask]
                    med = float(np.median(valid)) if len(valid) > 0 else 0.0
                    arr[b][nd_mask] = med

    # 3. Validate band order if band descriptions/names are available
    validated_arr = arr[:13]
    if band_names is not None and any(n for n in band_names):
        norm_names = [_normalize_band_name(n) if n else f"B{i+1:02d}" for i, n in enumerate(band_names)]
        # Check if canonical bands are all present in some order
        if all(cb in norm_names for cb in CANONICAL_SENTINEL2_BANDS):
            reordered_indices = [norm_names.index(cb) for cb in CANONICAL_SENTINEL2_BANDS]
            validated_arr = arr[reordered_indices]
        else:
            present = [n for n in norm_names if n in CANONICAL_SENTINEL2_BANDS]
            missing = [cb for cb in CANONICAL_SENTINEL2_BANDS if cb not in norm_names]
            if len(missing) > 0 and len(present) < 13:
                warnings.warn(
                    f"Band metadata incomplete or non-standard. Missing expected Sentinel-2 bands: {missing}. "
                    "Assuming default EuroSAT 13-band sequential indexing."
                )

    # 4. Validate reflectance scale against training data distribution
    max_val = float(np.max(validated_arr))
    min_val = float(np.min(validated_arr))

    if min_val < -500.0:
        raise ValueError(
            f"GeoTIFF contains invalid negative reflectance values (minimum: {min_val:.2f}). "
            "Please check radiometric calibration."
        )

    # If image is in unscaled surface reflectance [0.0, 1.0] (or [0.0, 2.0] for clouds/specular)
    if 0.0 <= max_val <= 2.0:
        warnings.warn(
            f"GeoTIFF appears to be in unscaled reflectance [0, 1] (max={max_val:.3f}). "
            "Rescaling by 10,000 to match EuroSAT Sentinel-2 Digital Number (DN) scale."
        )
        validated_arr = validated_arr * 10000.0
    elif max_val > 65535.0:
        raise ValueError(
            f"GeoTIFF reflectance scale exceeds 16-bit radiometric limits (max={max_val:.1f}). "
            "Data scale is incompatible with trained BDH weights."
        )

    return validated_arr


def pad_to_min_dimensions(
    arr: np.ndarray, min_h: int = 64, min_w: int = 64
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pads array if spatial dimensions are smaller than (min_h, min_w).
    Uses boundary edge repetition to preserve radiometric spectral properties without introducing arbitrary zeros.
    Returns:
        padded_arr: (C, max(H, min_h), max(W, min_w))
        original_shape: (H, W)
    """
    _, H, W = arr.shape
    pad_h = max(0, min_h - H)
    pad_w = max(0, min_w - W)
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        return padded, (H, W)
    return arr, (H, W)


def load_geotiff(
    file_or_bytes: Union[str, bytes, io.BytesIO],
    return_meta: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Reads and validates a Sentinel-2 GeoTIFF from path or in-memory bytes.
    Validates band count, order, reflectance scale, and inspects spatial resolution and CRS.
    
    If return_meta=True, returns (validated_arr, metadata_dict).
    If return_meta=False, returns validated_arr.
    """
    import rasterio
    if isinstance(file_or_bytes, (bytes, bytearray)):
        file_or_bytes = io.BytesIO(file_or_bytes)

    with rasterio.open(file_or_bytes) as src:
        arr = src.read().astype(np.float32)
        band_names = list(src.descriptions) if src.descriptions else None
        nodatavals = src.nodatavals
        crs_str = str(src.crs) if src.crs else None
        res_tuple = (float(src.res[0]), float(src.res[1])) if src.res else None

        meta = {
            "crs": crs_str,
            "res": res_tuple,
            "bounds": tuple(src.bounds) if src.bounds else None,
            "transform": list(src.transform) if src.transform else None,
            "descriptions": src.descriptions,
            "warnings": [],
        }

        # Validate spatial resolution & CRS
        if res_tuple is not None:
            rx, ry = res_tuple
            if src.crs and src.crs.is_geographic:
                meta["warnings"].append(
                    f"Dataset is in angular coordinates (degrees, {crs_str}). "
                    "EuroSAT training tiles use metric UTM projections (~10m/pixel ground sampling distance)."
                )
            elif rx > 0:
                if rx < 7.0 or rx > 15.0:
                    meta["warnings"].append(
                        f"Spatial resolution ({rx:.1f}m × {ry:.1f}m/pixel) differs from EuroSAT's native 10m/pixel Sentinel-2 resolution. "
                        "Receptive field scale mismatch may affect classification accuracy."
                    )
        else:
            meta["warnings"].append("Spatial resolution metadata missing from GeoTIFF.")

        if not crs_str:
            meta["warnings"].append("Coordinate Reference System (CRS) metadata missing from GeoTIFF.")

    validated = validate_geotiff_bands(arr, band_names=band_names, nodatavals=nodatavals)
    if return_meta:
        return validated, meta
    return validated


def load_rgb_image(file_or_bytes: Union[str, bytes, io.BytesIO], mean_vals: List[float]) -> np.ndarray:
    """
    Reads an RGB image (PNG, JPG) and prepares an EXPERIMENTAL synthetic 13-band Sentinel-2 representation:
    - Red -> Band 4 (B04, index 3), Green -> Band 3 (B03, index 2), Blue -> Band 2 (B02, index 1).
    - Estimates Near-Infrared B08 (index 7) and Narrow NIR B8A (index 8).
    - Fills other spectral bands (Red Edge, SWIR, etc.) with dataset training mean statistics.
    
    NOTE: This is strictly an experimental heuristic for qualitative UI demonstration.
    Predictions on ordinary RGB images lack genuine multi-spectral measurements and must not
    be treated as scientifically equivalent to calibrated 13-band Sentinel-2 observations.
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
    arr[3] = rgb[:, :, 0] * scale  # B04 (Red, index 3)
    arr[2] = rgb[:, :, 1] * scale  # B03 (Green, index 2)
    arr[1] = rgb[:, :, 2] * scale  # B02 (Blue, index 1)
    arr[7] = np.maximum(arr[3], arr[2]) * 1.3  # B08 (NIR proxy, index 7)
    arr[8] = arr[7] * 0.95                     # B8A (Narrow NIR proxy, index 8 in EuroSAT order)
    return arr


def extract_sliding_tiles(
    arr: np.ndarray, tile_size: int = 64, stride: int = 64
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], Tuple[int, int]]:
    """
    Splits an arbitrary-sized (13, H, W) satellite scene into 64x64 tiles for BDH.
    
    Robustness guarantees:
    - If H < 64 or W < 64, the scene is padded using boundary edge replication so
      a valid tile is always extracted and zero tiles are never returned.
    - Handles arbitrary spatial dimensions and edge tiles explicitly so edge pixels
      are never omitted.
      
    Returns:
        tiles: list of (13, 64, 64) numpy arrays
        coords: list of (y1, x1, y2, x2) bounding boxes in original scene coordinates
        (H, W): original spatial dimensions before padding
    """
    _, H, W = arr.shape
    work_arr, _ = pad_to_min_dimensions(arr, min_h=tile_size, min_w=tile_size)
    _, cur_H, cur_W = work_arr.shape

    if cur_H == tile_size and cur_W == tile_size:
        return [work_arr], [(0, 0, H, W)], (H, W)

    tiles = []
    coords = []

    # Calculate steps
    y_starts = list(range(0, cur_H - tile_size + 1, stride))
    if not y_starts or y_starts[-1] + tile_size < cur_H:
        y_starts.append(cur_H - tile_size)

    x_starts = list(range(0, cur_W - tile_size + 1, stride))
    if not x_starts or x_starts[-1] + tile_size < cur_W:
        x_starts.append(cur_W - tile_size)

    for y in y_starts:
        for x in x_starts:
            tile = work_arr[:, y : y + tile_size, x : x + tile_size]
            if tile.shape[1:] == (tile_size, tile_size):
                tiles.append(tile)
                # Store clipped boundary in original scene coordinate space
                coords.append((y, x, min(y + tile_size, H), min(x + tile_size, W)))

    return tiles, coords, (H, W)


def normalize_batch(tiles: List[np.ndarray], mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    """Converts a list of (13, 64, 64) numpy tiles into a normalized (B, 13, 64, 64) PyTorch tensor."""
    batch_np = np.stack(tiles, axis=0)  # (B, 13, 64, 64)
    x = torch.from_numpy(batch_np).float()
    return (x - mean_t) / (std_t + 1e-6)
