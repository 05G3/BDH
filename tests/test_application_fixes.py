import json
import os
import numpy as np
import pytest
import torch
import warnings

from application.inference import (
    load_bdh_model,
    predict_single_tile,
    predict_scene_sliding_window,
)
from application.preprocessing import (
    validate_geotiff_bands,
    pad_to_min_dimensions,
    extract_sliding_tiles,
    normalize_batch,
    load_rgb_image,
    load_geotiff,
    CANONICAL_SENTINEL2_BANDS,
)
import streamlit_app


def test_fix_1_missing_band_descriptions_warning():
    """Verify explicit warning when GeoTIFF lacks band descriptions."""
    arr = np.random.uniform(500, 3000, (13, 64, 64)).astype(np.float32)
    with pytest.warns(UserWarning, match="GeoTIFF is missing band descriptions"):
        validate_geotiff_bands(arr, band_names=None)

    # When metadata is requested from load_geotiff, warning should be captured
    _, meta = load_geotiff("demo_images/Forest_sample.tif", return_meta=True)
    assert any("Band descriptions are missing" in w for w in meta["warnings"])


def test_fix_1_canonical_band_order_and_b8a_position():
    """Verify standard EuroSAT-MS 13-band order: B8A is at index 8, followed by B09 at index 9."""
    expected_order = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
    ]
    assert CANONICAL_SENTINEL2_BANDS == expected_order
    assert CANONICAL_SENTINEL2_BANDS[7] == "B08"
    assert CANONICAL_SENTINEL2_BANDS[8] == "B8A"
    assert CANONICAL_SENTINEL2_BANDS[9] == "B09"

    # Test band reordering
    reversed_bands = list(reversed(CANONICAL_SENTINEL2_BANDS))
    raw_bands = np.zeros((13, 64, 64), dtype=np.float32)
    for i in range(13):
        raw_bands[i] = i * 100 + 500

    reordered = validate_geotiff_bands(raw_bands, band_names=reversed_bands)
    # Canonical index 8 (B8A) was at index 4 in reversed_bands (12 - 8 = 4)
    expected_b8a_val = 4 * 100 + 500
    assert np.allclose(reordered[8], expected_b8a_val)


def test_fix_2_radiometric_scaling_validation():
    """Verify radiometric scaling validations (all-zero, negative, unscaled [0,1], 8-bit, extreme values)."""
    # 1. All-zero array raises ValueError
    all_zero = np.zeros((13, 64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="all-zero reflectance values"):
        validate_geotiff_bands(all_zero)

    # 2. Negative values < -100 raise ValueError
    invalid_neg = np.full((13, 64, 64), -150.0, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid negative reflectance values"):
        validate_geotiff_bands(invalid_neg)

    # 3. Unscaled [0, 1] reflectance rescaled to DN ~10,000 with warning
    unscaled = np.full((13, 64, 64), 0.25, dtype=np.float32)
    with pytest.warns(UserWarning, match="unscaled reflectance"):
        scaled = validate_geotiff_bands(unscaled)
    assert np.isclose(scaled[0, 0, 0], 2500.0)

    # 4. 8-bit scale warning ([0, 255])
    eight_bit = np.full((13, 64, 64), 200.0, dtype=np.float32)
    with pytest.warns(UserWarning, match="max pixel value is 200.0 <= 255.0"):
        validate_geotiff_bands(eight_bit)

    # 5. Values > 25,000 raise ValueError
    extreme_high = np.full((13, 64, 64), 30000.0, dtype=np.float32)
    with pytest.raises(ValueError, match="exceed valid Sentinel-2 range"):
        validate_geotiff_bands(extreme_high)


def test_fix_3_streamlit_arr_variable_reference():
    """Verify streamlit_app.py uses input_array.shape rather than undefined arr.shape in tile info."""
    with open("streamlit_app.py", "r") as f:
        lines = f.readlines()
    # Check that in Case B (single tile preview), the info box uses input_array.shape[0], NOT arr.shape[0]
    single_tile_lines = [l for l in lines if "padded to 64×64 for BDH receptive field" in l]
    assert len(single_tile_lines) > 0
    for l in single_tile_lines:
        assert "{input_array.shape[0]} bands" in l
        assert "{arr.shape[0]} bands" not in l


def test_fix_4_spatial_resolution_descriptions():
    """Verify spatial resolution is not assumed to be 10m/pixel for all arbitrary inputs."""
    with open("streamlit_app.py", "r") as f:
        content = f.read()
    # Should not assert every image preserves 10m/pixel
    assert "preserve native 10m/pixel spatial resolution" not in content
    # Checks for spatial_res_desc
    assert "spatial_res_desc" in content


def test_small_image_and_edge_tiles():
    """Verify small images (< 64x64) and edge dimensions never return 0 tiles."""
    small_arr = np.random.uniform(500, 3000, (13, 32, 32)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(small_arr, tile_size=64, stride=64)
    assert len(tiles) == 1
    assert tiles[0].shape == (13, 64, 64)
    assert orig_shape == (32, 32)
    assert coords[0] == (0, 0, 32, 32)


def test_full_inference_with_actual_test_data():
    """Verify genuine model inference on an actual test tile and demo GeoTIFF."""
    model, meta = load_bdh_model("results/checkpoints/bdh_main_best.pt", torch.device("cpu"))
    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean_t = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std_t = torch.tensor(stats["std"]).view(1, 13, 1, 1)

    forest_arr = load_geotiff("demo_images/Forest_sample.tif")
    x = normalize_batch([forest_arr], mean_t, std_t)
    res = predict_single_tile(model, x, torch.device("cpu"), record_stats=True)

    assert res["pred_idx"] == 1, f"Expected Forest (1), got {res['pred_idx']}"
    assert res["confidence"] > 0.95
    assert len(res["sparsity_stats"]) == 4
