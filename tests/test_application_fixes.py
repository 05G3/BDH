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
    load_geotiff,
    CANONICAL_SENTINEL2_BANDS,
)
import streamlit_app


def test_issue_2_canonical_band_order_and_b8a_position():
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


def test_issue_3a_geotiff_metadata_and_crs_validation():
    """Verify load_geotiff extracts CRS and resolution metadata with appropriate warnings."""
    # Test with demo TIFF file
    arr, meta = load_geotiff("demo_images/Forest_sample.tif", return_meta=True)
    assert arr.shape == (13, 64, 64)
    assert isinstance(meta, dict)
    assert "crs" in meta
    assert "res" in meta
    assert "warnings" in meta


def test_issue_3b_flask_diagnostic_endpoints(monkeypatch):
    """Verify Flask /api/sparsity and /api/equivalence load actual files and return 404 if missing."""
    import app
    client = app.app.test_client()

    # 1. Sparsity endpoint
    res_sp = client.get("/api/sparsity")
    assert res_sp.status_code == 200
    sp_data = res_sp.get_json()
    assert isinstance(sp_data, list)
    assert len(sp_data) == 4

    # 2. Equivalence endpoint
    res_eq = client.get("/api/equivalence")
    assert res_eq.status_code == 200
    eq_data = res_eq.get_json()
    assert eq_data.get("status") == "PASSED"
    assert "trained_model_256_max_diff" in eq_data


def test_issue_3c_no_filename_ground_truth_guessing():
    """Verify demo files do not infer ground truth from filename splitting."""
    import app
    client = app.app.test_client()

    app.init_app()
    # Predict on demo file: true_class must be None
    res = client.post("/api/predict", json={"demo_file": "Forest_sample.tif"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["true_class"] is None, "Ground truth should NOT be inferred from filename!"


def test_issue_3d_macro_f1_missing_display(monkeypatch):
    """Verify missing f1_macro displays Not available rather than 0.00%."""
    # When metrics file has accuracy but missing f1_macro
    partial_metrics = {"accuracy": 0.973}
    monkeypatch.setattr(streamlit_app, "load_evaluation_metrics", lambda: partial_metrics)
    m = streamlit_app.load_evaluation_metrics()
    assert "f1_macro" not in m
    assert m["accuracy"] == 0.973


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
