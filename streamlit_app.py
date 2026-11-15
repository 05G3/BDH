import io
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from application.inference import (
    load_bdh_model,
    predict_single_tile,
    predict_scene_sliding_window,
)
from application.preprocessing import (
    load_geotiff,
    load_rgb_image,
    extract_sliding_tiles,
    normalize_batch,
    pad_to_min_dimensions,
)
from application.visualization import (
    CLASS_PALETTE,
    get_rgb_preview,
    get_nir_preview,
    render_landcover_map,
    create_segmented_overlay,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BDH Satellite Land-Cover Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "results/checkpoints/bdh_main_best.pt"
STATS_PATH = "data/splits/norm_stats.json"
TRAIN_CSV = "data/splits/train.csv"
TEST_IMAGES_PATH = "data/processed/test_images.npy"
TEST_LABELS_PATH = "data/processed/test_labels.npy"
METRICS_PATHS = [
    "results/tables/bdh_main_test_metrics.json",
    "results/tables/bdh_main_s0_test_metrics.json",
]

DEFAULT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Initialize Session State
if "input_array" not in st.session_state:
    st.session_state.input_array = None
if "input_source_desc" not in st.session_state:
    st.session_state.input_source_desc = ""
if "ground_truth_class" not in st.session_state:
    st.session_state.ground_truth_class = None
if "is_multi_tile_scene" not in st.session_state:
    st.session_state.is_multi_tile_scene = False
if "is_synthetic_rgb" not in st.session_state:
    st.session_state.is_synthetic_rgb = False
if "scene_results" not in st.session_state:
    st.session_state.scene_results = None
if "single_results" not in st.session_state:
    st.session_state.single_results = None

# -----------------------------------------------------------------------------
# Cached Resources (Model, Stats, Classes, Test Samples, Metrics)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the trained BDH model."""
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at '{CKPT_PATH}'.")
    return load_bdh_model(CKPT_PATH, DEVICE)


@st.cache_data(show_spinner=False)
def load_normalization_stats() -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Loads mean and standard deviation per Sentinel-2 band."""
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
    mean_t = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 13, 1, 1)
    std_t = torch.tensor(stats["std"], dtype=torch.float32).view(1, 13, 1, 1)
    return mean_t, std_t, stats["mean"]


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Optional[Dict[str, Any]]:
    """
    Dynamically loads evaluation metrics from actual results/tables JSON files.
    Returns None if verified evaluation files do not exist (no hardcoded fallback numbers).
    """
    for p in METRICS_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
                    metrics["source_file"] = p
                    return metrics
            except Exception:
                pass
    return None


@st.cache_data(show_spinner=False)
def load_sample_library():
    """Extracts real Sentinel-2 test tiles for each of the 10 classes."""
    samples = {}
    if os.path.exists(TEST_IMAGES_PATH) and os.path.exists(TEST_LABELS_PATH):
        try:
            images = np.load(TEST_IMAGES_PATH, mmap_mode="r")
            labels = np.load(TEST_LABELS_PATH)
            for c_idx in range(len(DEFAULT_CLASSES)):
                matches = np.where(labels == c_idx)[0]
                if len(matches) > 0:
                    idx = int(matches[0])
                    samples[DEFAULT_CLASSES[c_idx]] = {
                        "array": np.array(images[idx]),
                        "label_idx": c_idx,
                        "class_name": DEFAULT_CLASSES[c_idx],
                        "test_idx": idx,
                    }
        except Exception:
            pass
    return samples


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    try:
        model, model_meta = get_cached_model()
        mean_t, std_t, mean_list = load_normalization_stats()
        eval_metrics = load_evaluation_metrics()
        sample_library = load_sample_library()
        class_names = DEFAULT_CLASSES
    except Exception as e:
        st.error(f"Error loading BDH model or resources: {e}")
        st.stop()

    # Sidebar: Model Architecture & Evaluation Status
    with st.sidebar:
        st.title("🛰️ BDH Model Status")
        st.success("Trained Checkpoint Active")
        st.caption(f"Loaded: `{CKPT_PATH}`")

        # Display actual evaluation metrics or explicit 'Not available'
        if eval_metrics and "accuracy" in eval_metrics:
            acc_val = eval_metrics["accuracy"]
            f1_val = eval_metrics.get("f1_macro")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Test Accuracy", f"{acc_val * 100.0:.2f}%")
            with col_m2:
                if f1_val is not None and isinstance(f1_val, (int, float)):
                    st.metric("Macro-F1", f"{f1_val * 100.0:.2f}%")
                else:
                    st.metric("Macro-F1", "Not available")
            st.caption(f"Metrics Source: `{eval_metrics.get('source_file')}`")
        else:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Test Accuracy", "Not available")
            with col_m2:
                st.metric("Macro-F1", "Not available")
            st.caption("No precomputed metrics JSON file found.")

        st.metric("Trained Parameters", f"{model_meta['n_params']:,}")
        st.metric("Architecture", "BDH-GPU (Non-Causal)")
        st.metric("Device", str(DEVICE).upper())

        st.markdown("---")
        st.markdown("### 📋 10 Land-Cover Classes")
        for idx, c_name in enumerate(class_names):
            color_rgb = CLASS_PALETTE.get(idx, (100, 100, 100))
            hex_c = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            st.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;background-color:{hex_c};border-radius:2px;margin-right:6px;'></span> {c_name}",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            """
            **🛰️ Spectral Band Handling:**
            - **13-Band GeoTIFFs (`.tif`)**: Recommended primary input. Uses calibrated multi-spectral signatures (B01–B12 + B8A).
            - **Large Satellite Scenes**: Decomposed via sliding window ($64\\times 64$) to generate a **tile-based land-cover classification map**.
            - **RGB Images (`.png`, `.jpg`)**: Experimental demo only; non-RGB bands are synthetically approximated from dataset statistics.
            """
        )

    # Main Body Header
    st.title("🛰️ Sentinel-2 Satellite Land-Cover Classification")
    st.markdown(
        "Upload a Sentinel-2 satellite image or choose from benchmark test tiles to run live inference "
        "using your **actual trained BDH (Bio-Data-Hybrid) neural network**."
    )

    # Input Options Tabs
    tab_samples, tab_upload, tab_demo_files = st.tabs([
        "📚 Benchmark Test Tiles (Recommended)",
        "📤 Upload Satellite GeoTIFF / RGB Image",
        "📁 Pre-Packaged Demo Scenes",
    ])

    # 1. Benchmark Test Tiles Tab
    with tab_samples:
        st.markdown("Select a ground-truth satellite tile from the EuroSAT held-out test split:")
        if sample_library:
            col_sel, col_btn = st.columns([3, 1])
            with col_sel:
                selected_class = st.selectbox(
                    "Choose Land-Cover Class:",
                    list(sample_library.keys()),
                    index=1,  # Default to Forest
                )
            with col_btn:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if st.button("📥 Load Benchmark Tile", key="btn_load_sample", use_container_width=True):
                    sample_data = sample_library[selected_class]
                    st.session_state.input_array = sample_data["array"]
                    st.session_state.ground_truth_class = sample_data["class_name"]
                    st.session_state.is_multi_tile_scene = False
                    st.session_state.is_synthetic_rgb = False
                    st.session_state.spatial_res_desc = "10.0m/pixel (EuroSAT native Sentinel-2)"
                    st.session_state.scene_results = None
                    st.session_state.single_results = None
                    st.session_state.input_source_desc = (
                        f"EuroSAT Test Split Tile #{sample_data['test_idx']} (Ground Truth: **{sample_data['class_name']}**)"
                    )
        else:
            st.info("Test dataset array not found in `data/processed/test_images.npy`.")

    # 2. Upload Tab
    with tab_upload:
        st.markdown(
            "Upload a multi-spectral GeoTIFF (`.tif`, `.tiff`) or standard image (`.png`, `.jpg`):"
        )
        uploaded_file = st.file_uploader(
            "Select satellite image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            fname = uploaded_file.name.lower()
            try:
                if fname.endswith((".tif", ".tiff")):
                    arr, meta = load_geotiff(file_bytes, return_meta=True)
                    st.session_state.is_synthetic_rgb = False
                    meta_res = meta.get("res")
                    if meta_res and meta_res[0] > 0 and (meta.get("crs") and not meta["crs"].startswith("EPSG:4326")):
                        st.session_state.spatial_res_desc = f"{meta_res[0]:.1f}m/pixel"
                    else:
                        st.session_state.spatial_res_desc = "unspecified in metadata (assumed ~10m/pixel scale)"
                    if meta.get("warnings"):
                        for w in meta["warnings"]:
                            st.warning(f"⚠️ GeoTIFF Metadata Notice: {w}")
                else:
                    arr = load_rgb_image(file_bytes, mean_list)
                    st.session_state.is_synthetic_rgb = True
                    st.session_state.spatial_res_desc = "uncalibrated RGB pixels (not Sentinel-2 GSD)"
                    st.warning(
                        "⚠️ **Experimental Demonstration Mode (Synthetic Spectral Bands)**: "
                        "You uploaded a 3-band RGB image. Synthetic bands (including NIR and SWIR) were mathematically approximated "
                        "from RGB channels and EuroSAT training statistics. For scientifically rigorous evaluation, upload genuine "
                        "13-band Sentinel-2 GeoTIFF files or select from the benchmark test tiles."
                    )

                _, H, W = arr.shape
                is_multi = (H > 64 or W > 64)
                st.session_state.input_array = arr
                st.session_state.ground_truth_class = None
                st.session_state.is_multi_tile_scene = is_multi
                st.session_state.scene_results = None
                st.session_state.single_results = None
                
                type_desc = "Synthetic RGB Proxy" if st.session_state.is_synthetic_rgb else "Genuine Multi-Spectral Sentinel-2"
                st.session_state.input_source_desc = (
                    f"Uploaded File: `{uploaded_file.name}` • {arr.shape[0]} bands, {H}×{W}px • Type: {type_desc}"
                )
                if H < 64 or W < 64:
                    st.info(
                        f"Spatial dimensions ({H}×{W}) are smaller than BDH's 64×64 receptive field. "
                        "Edge reflection padding will be automatically applied for model inference."
                    )
                st.success(f"File loaded successfully! Spatial dimensions: {H}×{W} pixels ({arr.shape[0]} bands).")
            except Exception as ex:
                st.error(f"GeoTIFF Validation Error: {ex}")

    # 3. Demo Scenes Tab
    with tab_demo_files:
        st.markdown("Choose from pre-packaged Sentinel-2 demonstration files in `demo_images/`:")
        demo_files = [f for f in os.listdir("demo_images") if f.endswith(".tif")] if os.path.exists("demo_images") else []
        if demo_files:
            col_d_sel, col_d_btn = st.columns([3, 1])
            with col_d_sel:
                chosen_demo = st.selectbox("Select Demo File:", sorted(demo_files))
            with col_d_btn:
                st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
                if st.button("📥 Load Demo File", key="btn_load_demo", use_container_width=True):
                    arr, meta = load_geotiff(os.path.join("demo_images", chosen_demo), return_meta=True)
                    _, H, W = arr.shape
                    is_multi = (H > 64 or W > 64)
                    st.session_state.input_array = arr
                    st.session_state.is_multi_tile_scene = is_multi
                    st.session_state.is_synthetic_rgb = False
                    st.session_state.ground_truth_class = None  # Verified ground truth not inferred from filename
                    st.session_state.spatial_res_desc = "10.0m/pixel (EuroSAT native Sentinel-2)"
                    st.session_state.scene_results = None
                    st.session_state.single_results = None

                    if is_multi:
                        st.session_state.input_source_desc = (
                            f"Demo Multi-Tile Scene: `{chosen_demo}` ({H}×{W}px, {arr.shape[0]} bands)"
                        )
                    else:
                        st.session_state.input_source_desc = (
                            f"Demo Tile: `{chosen_demo}` ({H}×{W}px, {arr.shape[0]} bands)"
                        )
                    if meta.get("warnings"):
                        for w in meta["warnings"]:
                            st.info(f"ℹ️ {w}")
                    st.success(f"Loaded {chosen_demo} ({H}×{W}px)!")

    # -------------------------------------------------------------------------
    # Visualization & Inference Section
    # -------------------------------------------------------------------------
    if st.session_state.input_array is not None:
        input_array = st.session_state.input_array
        is_multi_tile = st.session_state.is_multi_tile_scene
        is_synth = st.session_state.is_synthetic_rgb
        _, H, W = input_array.shape

        st.markdown("---")

        if is_synth:
            st.warning(
                "🧪 **Notice**: Active input is a synthetic RGB approximation. Results should be viewed as an interactive interface demo rather than validated Sentinel-2 radiometric predictions."
            )

        # CASE A: Large Multi-Tile Satellite Scene (Sliding-Window Land-Cover Mapping)
        if is_multi_tile:
            st.subheader(f"🗺️ Multi-Tile Satellite Scene ({H}×{W} pixels)")
            st.caption(st.session_state.input_source_desc)

            res_desc = st.session_state.get("spatial_res_desc", "assumed ~10m/pixel")
            st.info(
                "ℹ️ **Spatial Mapping Note**: BDH is a patch-level classifier trained on 64×64 EuroSAT satellite tiles. "
                "EuroSAT training tiles were acquired at native Sentinel-2 resolution (10m ground sampling distance, yielding a "
                "640m×640m footprint per tile). The output across large scenes is a **tile-based land-cover classification map** "
                "(each 64×64 patch displays its dominant predicted class), **not** a pixel-wise semantic segmentation map.\n\n"
                "**Boundary Overlap Notice**: When scene dimensions are not an exact multiple of the 64px stride, "
                "the boundary tile shifts backward to fully cover edge pixels. Predictions in overlap zones represent patch-level categorical "
                "aggregations, not pixel-level evidence or exact geodetic area measurements."
            )

            tiles, coords, orig_shape = extract_sliding_tiles(input_array, tile_size=64, stride=64)
            st.markdown(
                f"**Sliding Window Pipeline:** Decomposed {H}×{W} scene into **{len(tiles)} standard 64×64 patches** "
                f"(stride 64px) to match BDH's 64×64 receptive field without spatial resampling. "
                f"(Input resolution: **{res_desc}**)."
            )

            col_s_left, col_s_right = st.columns([1, 1], gap="medium")

            rgb_scene = get_rgb_preview(input_array)
            nir_scene = get_nir_preview(input_array)

            with col_s_left:
                st.image(
                    rgb_scene,
                    caption=f"True-Color RGB Satellite Scene ({H}×{W})",
                    use_container_width=True,
                )

            with col_s_right:
                st.image(
                    nir_scene,
                    caption="False-Color Infrared Composite (NIR B08, Red B04, Green B03)",
                    use_container_width=True,
                )

            st.markdown("### Model Scene Inference")
            if st.button("🚀 Run BDH Land-Cover Mapping", type="primary", use_container_width=True, key="btn_scene_infer"):
                with st.spinner(f"Classifying {len(tiles)} patches with BDH neural network..."):
                    scene_res = predict_scene_sliding_window(
                        model=model,
                        tiles=tiles,
                        coords=coords,
                        orig_shape=orig_shape,
                        mean_t=mean_t,
                        std_t=std_t,
                        device=DEVICE,
                        class_names=class_names,
                    )
                    st.session_state.scene_results = scene_res

            if st.session_state.scene_results is not None:
                res = st.session_state.scene_results
                tile_map = res["tile_map"]

                st.success(
                    f"Inference completed in **{res['total_latency_ms']:.1f} ms** across **{res['total_tiles']} patches** "
                    f"({res['total_latency_ms']/max(1, res['total_tiles']):.2f} ms/tile on {DEVICE})!"
                )

                tab_map1, tab_map2 = st.tabs([
                    "🗺️ Colorized Tile-Based Land-Cover Map",
                    "🛰️ Blended Overlay on Satellite Imagery"
                ])

                with tab_map1:
                    col_m1, col_m2 = st.columns([1.2, 1])
                    with col_m1:
                        map_img = render_landcover_map(tile_map)
                        st.image(
                            map_img,
                            caption="BDH Tile-Based Land-Cover Classification Map (64×64 Patch-Level Aggregation)",
                            use_container_width=True,
                        )
                    with col_m2:
                        st.markdown("#### Land-Cover Composition Breakdown:")
                        if res["composition"]:
                            comp_df = pd.DataFrame([
                                {"Class": k, "Area Coverage": f"{v:.1f}%", "Percentage": v}
                                for k, v in res["composition"].items()
                            ]).sort_values("Percentage", ascending=False)
                            st.dataframe(comp_df[["Class", "Area Coverage"]].reset_index(drop=True), use_container_width=True)
                            st.bar_chart(comp_df.set_index("Class")["Percentage"], color="#0A6E6E")
                        else:
                            st.write("No classified pixels.")

                with tab_map2:
                    alpha_val = st.slider("Classification Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
                    overlay_img = create_segmented_overlay(rgb_scene, tile_map, alpha=alpha_val)
                    st.image(
                        overlay_img,
                        caption=f"Blended Overlay ({int(alpha_val*100)}% Tile Classification + True-Color Imagery)",
                        use_container_width=True,
                    )

        # CASE B: Single Satellite Tile (Standard Benchmark Workflow)
        else:
            col_img, col_pred = st.columns([1, 1.25], gap="large")

            # Column 1: Image Preprocessing & Spectral Inspection
            with col_img:
                st.subheader("Step 1 & 2: Preprocessing")
                st.caption(st.session_state.input_source_desc)

                rgb_preview = get_rgb_preview(input_array)
                st.image(
                    rgb_preview,
                    caption="True-Color RGB Composite (B04, B03, B02)",
                    use_container_width=True,
                )

                nir_preview = get_nir_preview(input_array)
                with st.expander("Show False-Color Infrared (Vegetation NIR B08)"):
                    st.image(
                        nir_preview,
                        caption="False-Color NIR Composite (B08, B04, B03)",
                        use_container_width=True,
                    )

                if H < 64 or W < 64:
                    st.info(f"Tile dimensions: `{input_array.shape[0]} bands, {H}×{W}px` (padded to 64×64 for BDH receptive field)")
                else:
                    st.info(f"Tensor Dimensions: `(13, {H}, {W})` • Patches: 64 ($8\\times 8$ grid) • Embedding: 256-dim")

            # Column 2: Live BDH Model Inference
            with col_pred:
                st.subheader("Step 3 & 4: Model Inference")

                predict_clicked = st.button("🚀 Predict Land-Cover Class", type="primary", use_container_width=True, key="btn_single_infer")

                if predict_clicked:
                    # Pad to minimum receptive field if smaller than 64x64
                    padded_arr, _ = pad_to_min_dimensions(input_array, min_h=64, min_w=64)
                    norm_tensor = normalize_batch([padded_arr], mean_t, std_t)
                    pred_res = predict_single_tile(
                        model=model,
                        normalized_tensor=norm_tensor,
                        device=DEVICE,
                        record_stats=True,
                    )
                    st.session_state.single_results = pred_res

                if st.session_state.single_results is not None:
                    res = st.session_state.single_results
                    probs = res["probabilities"]
                    pred_idx = res["pred_idx"]
                    predicted_class = class_names[pred_idx]
                    confidence = res["confidence"]
                    latency_ms = res["latency_ms"]
                    sparsity_stats = res.get("sparsity_stats", [])

                    gt = st.session_state.ground_truth_class
                    is_correct = (predicted_class == gt) if gt else None
                    status_color = "#0A6E6E" if (is_correct is None or is_correct) else "#B91C1C"

                    synth_badge = (
                        "<div style='display:inline-block;background:#FEF3C7;color:#B45309;"
                        "padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700;"
                        "margin-bottom:8px;'>EXPERIMENTAL SYNTHETIC INPUT</div>"
                        if is_synth else ""
                    )
                    gt_badge = (
                        f"<div style='color:{status_color};font-weight:700;font-size:0.95rem;margin-top:8px;'>"
                        f"{'✅ Correct Prediction' if is_correct else '⚠️ Ground truth: ' + gt}</div>"
                        if gt else ""
                    )

                    # Output Card rendered directly without markdown indentation
                    output_html = (
                        f"<div style='background-color:{status_color}14;border:2.5px solid {status_color};"
                        f"border-radius:12px;padding:22px;text-align:center;margin-bottom:20px;'>"
                        f"{synth_badge}"
                        f"<div style='font-size:0.88rem;text-transform:uppercase;letter-spacing:0.05em;"
                        f"color:{status_color};font-weight:700;'>Predicted Land-Use Class</div>"
                        f"<h1 style='margin:8px 0;font-size:2.6rem;color:{status_color};'>{predicted_class}</h1>"
                        f"<div style='font-size:1.25rem;font-weight:700;'>Prediction Confidence: {confidence*100:.1f}%</div>"
                        f"<div style='font-size:0.85rem;color:#666;margin-top:4px;'>"
                        f"Inference Latency: {latency_ms:.2f} ms ({DEVICE})</div>"
                        f"{gt_badge}"
                        f"</div>"
                    )

                    if hasattr(st, "html"):
                        st.html(output_html)
                    else:
                        st.markdown(output_html, unsafe_allow_html=True)

                    # Class Probabilities Bar Chart
                    st.markdown("#### Softmax Class Probabilities:")
                    prob_df = pd.DataFrame({
                        "Class": class_names,
                        "Probability": probs * 100.0,
                    }).sort_values("Probability", ascending=False)

                    st.bar_chart(prob_df.set_index("Class"), color="#0A6E6E")

                    # Top-3 Predictions Table
                    st.markdown("#### Top-3 Predictions:")
                    top3_df = prob_df.head(3).copy()
                    top3_df["Probability"] = top3_df["Probability"].map(lambda p: f"{p:.2f}%")
                    st.table(top3_df.reset_index(drop=True))

                    # BDH Internal Sparsity Diagnostics Expander
                    if sparsity_stats:
                        with st.expander("🔬 BDH Internal Activation Sparsity Diagnostics"):
                            st.markdown(
                                "Fraction of exactly-zero activations after ReLU across the 4 weight-tied BDH levels:"
                            )
                            sparse_rows = []
                            for lvl, s in enumerate(sparsity_stats, start=1):
                                sparse_rows.append({
                                    "Level": f"Level {lvl}",
                                    "x_sparse (First Code)": f"{s['x_sparse']*100:.1f}%",
                                    "y_sparse (Attended)": f"{s['y_sparse']*100:.1f}%",
                                    "xy_sparse (Gated Code)": f"{s['xy_sparse']*100:.1f}%",
                                    })
                            st.dataframe(pd.DataFrame(sparse_rows).set_index("Level"))
                            st.caption(
                                "Multiplicative gating ($u \\odot v$) drives activation sparsity to ~92%–93% zeros before decoding."
                            )

    st.markdown("---")
    st.caption(
        "BDH Sentinel-2 Land-Cover Classification • EuroSAT-MS 13-band Benchmark • "
        "Model Checkpoint: `results/checkpoints/bdh_main_best.pt`"
    )


if __name__ == "__main__":
    main()
