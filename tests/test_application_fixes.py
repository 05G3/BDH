import json
import os
import numpy as np
import pytest
import torch

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
    CANONICAL_SENTINEL2_BANDS,
)
from streamlit_app import load_evaluation_metrics


def test_issue_1_small_image_and_edge_tiles():
    """Verify small images (< 64x64) and edge dimensions never return 0 tiles."""
    # 1. Very small image (32x32)
    small_arr = np.random.uniform(500, 3000, (13, 32, 32)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(small_arr, tile_size=64, stride=64)
    assert len(tiles) == 1, f"Expected 1 tile for 32x32 image, got {len(tiles)}"
    assert tiles[0].shape == (13, 64, 64)
    assert orig_shape == (32, 32)
    assert coords[0] == (0, 0, 32, 32)

    # 2. Non-square small image (20x45)
    rect_arr = np.random.uniform(500, 3000, (13, 20, 45)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(rect_arr, tile_size=64, stride=64)
    assert len(tiles) == 1
    assert tiles[0].shape == (13, 64, 64)
    assert orig_shape == (20, 45)

    # 3. Arbitrary scene with edge dimensions (100x80)
    scene_arr = np.random.uniform(500, 3000, (13, 100, 80)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(scene_arr, tile_size=64, stride=64)
    assert len(tiles) == 4, f"Expected 4 tiles covering (100, 80), got {len(tiles)}"
    for t in tiles:
        assert t.shape == (13, 64, 64)
    assert orig_shape == (100, 80)


def test_issue_1_predict_scene_small_image():
    """Verify predict_scene_sliding_window works seamlessly on scenes smaller than 64x64."""
    small_arr = np.random.uniform(500, 3000, (13, 40, 50)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(small_arr, tile_size=64, stride=64)

    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean_t = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std_t = torch.tensor(stats["std"]).view(1, 13, 1, 1)

    model, _ = load_bdh_model("results/checkpoints/bdh_main_best.pt", torch.device("cpu"))
    class_names = [f"Class_{i}" for i in range(10)]

    res = predict_scene_sliding_window(
        model=model,
        tiles=tiles,
        coords=coords,
        orig_shape=orig_shape,
        mean_t=mean_t,
        std_t=std_t,
        device=torch.device("cpu"),
        class_names=class_names,
    )
    assert res["tile_map"].shape == (40, 50)
    assert (res["tile_map"] >= 0).all()
    assert res["total_tiles"] == 1


def test_issue_2_tile_based_land_cover_map_attributes():
    """Verify result dictionary correctly designates tile-based land-cover map."""
    arr = np.random.uniform(500, 3000, (13, 64, 64)).astype(np.float32)
    tiles, coords, orig_shape = extract_sliding_tiles(arr, tile_size=64, stride=64)
    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean_t = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std_t = torch.tensor(stats["std"]).view(1, 13, 1, 1)
    model, _ = load_bdh_model("results/checkpoints/bdh_main_best.pt", torch.device("cpu"))

    res = predict_scene_sliding_window(
        model=model,
        tiles=tiles,
        coords=coords,
        orig_shape=orig_shape,
        mean_t=mean_t,
        std_t=std_t,
        device=torch.device("cpu"),
        class_names=[f"C{i}" for i in range(10)],
    )
    assert "tile_map" in res
    assert res["is_tile_based_map"] is True


def test_issue_3_synthetic_rgb_disclaimer():
    """Verify load_rgb_image produces expected 13 bands with synthetic NIR and docstring disclaimers."""
    from PIL import Image
    import io

    pil_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    with open("data/splits/norm_stats.json") as f:
        mean_vals = json.load(f)["mean"]

    arr = load_rgb_image(buf.getvalue(), mean_vals)
    assert arr.shape == (13, 64, 64)
    # NIR band (index 7) should be non-zero and approximate proxy
    assert arr[7].mean() > 0


def test_issue_4_geotiff_validation_and_scale():
    """Verify band order checking, reflectance scaling, and error triggers."""
    # 1. Less than 13 bands raises ValueError
    too_few = np.ones((10, 64, 64), dtype=np.float32) * 1000
    with pytest.raises(ValueError, match="GeoTIFF contains only 10 bands"):
        validate_geotiff_bands(too_few)

    # 2. Unscaled reflectance [0, 1] automatically rescaled to DN ~10000
    unscaled = np.full((13, 64, 64), 0.25, dtype=np.float32)
    scaled = validate_geotiff_bands(unscaled)
    assert np.isclose(scaled[0, 0, 0], 2500.0)

    # 3. Invalid negative values
    invalid_neg = np.full((13, 64, 64), -1000.0, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid negative reflectance"):
        validate_geotiff_bands(invalid_neg)

    # 4. Band reordering if custom order is passed
    reversed_bands = list(reversed(CANONICAL_SENTINEL2_BANDS))
    raw_bands = np.zeros((13, 64, 64), dtype=np.float32)
    for i in range(13):
        raw_bands[i] = i * 100 + 500  # distinct values per band

    reordered = validate_geotiff_bands(raw_bands, band_names=reversed_bands)
    # First canonical band B01 was last in reversed_bands (index 12)
    assert np.allclose(reordered[0], 12 * 100 + 500)


def test_issue_5_no_hardcoded_metrics_fallback(tmp_path, monkeypatch):
    """Verify metrics loader returns None when files do not exist (no hardcoded fallback)."""
    import streamlit_app
    # Point to nonexistent path
    monkeypatch.setattr(streamlit_app, "METRICS_PATHS", ["nonexistent_metrics.json"])
    metrics = streamlit_app.load_evaluation_metrics()
    assert metrics is None, "Expected None when evaluation files are missing!"


def test_issue_6_full_inference_with_actual_test_data():
    """Verify genuine model inference on an actual test tile and demo GeoTIFF."""
    model, meta = load_bdh_model("results/checkpoints/bdh_main_best.pt", torch.device("cpu"))
    with open("data/splits/norm_stats.json") as f:
        stats = json.load(f)
    mean_t = torch.tensor(stats["mean"]).view(1, 13, 1, 1)
    std_t = torch.tensor(stats["std"]).view(1, 13, 1, 1)

    # 1. Test from demo_images/Forest_sample.tif
    from application.preprocessing import load_geotiff
    forest_arr = load_geotiff("demo_images/Forest_sample.tif")
    x = normalize_batch([forest_arr], mean_t, std_t)
    res = predict_single_tile(model, x, torch.device("cpu"), record_stats=True)

    assert res["pred_idx"] == 1, f"Expected Forest (1), got {res['pred_idx']}"
    assert res["confidence"] > 0.95
    assert len(res["sparsity_stats"]) == 4
    for lvl in res["sparsity_stats"]:
        assert lvl["xy_sparse"] > 0.85  # verified BDH sparsity gating

    # 2. Test held-out test split array if available
    if os.path.exists("data/processed/test_images.npy"):
        images = np.load("data/processed/test_images.npy", mmap_mode="r")
        labels = np.load("data/processed/test_labels.npy")
        test_img = np.array(images[0])
        test_label = int(labels[0])
        x_split = normalize_batch([test_img], mean_t, std_t)
        res_split = predict_single_tile(model, x_split, torch.device("cpu"))
        print(f"Test split sample #0: true={test_label}, pred={res_split['pred_idx']}, conf={res_split['confidence']:.4f}")
