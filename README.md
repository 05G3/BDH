# BDH-inspired classification of EuroSAT-MS (13 bands)

## Setup
pip install -r requirements.txt

## Data
Download EuroSAT-MS (GeoTIFF) and place it at data/raw/EuroSAT_MS/<class>/*.tif

## Pipeline
python -m experiments.inspect_data
python -m experiments.prepare_data     # stratified 70/15/15 split (seed 42), train-only band stats
python -m experiments.cache_dataset    # builds data/processed/*.npy
python -m experiments.test_model
python -m experiments.test_equivalence

## Train / evaluate
python -m experiments.train_bdh --name bdh_main --model bdh --seed 0 --epochs 30 --weight_decay 0.05
python -m experiments.evaluate --name bdh_main
python -m experiments.run_suite --tier 1 2 3      # all baselines/ablations, 3 seeds
python -m experiments.aggregate_results
python -m experiments.measure_sparsity --name bdh_main_s0
python -m experiments.visualize_predictions --name bdh_main_s0

## Model
Patch embedding (832 -> 256) -> 4 tied-weight levels of: sparse ReLU projection,
non-softmax bidirectional associative attention (scores form $(QK^T)V$ with RoPE), gating, low-rank decoder,
residual LayerNorm -> GAP -> Linear(256, 10).
Adapted from Pathway's BDH-GPU (arXiv:2509.26507, github.com/pathwaycom/bdh);
differences: non-causal attention across 2D spatial patches, patch embedding and classification head instead of byte LM.

## Interactive Application & Live Showcase
Run the interactive Streamlit land-cover classifier:
```bash
streamlit run streamlit_app.py
```
Or launch the Flask demonstration dashboard:
```bash
python app.py --port 5050
```

### Features:
- **Benchmark Test Split**: Test tiles with known ground truth from EuroSAT-MS.
- **Multi-Spectral GeoTIFFs**: Supports full 13-band Sentinel-2 inputs (B01–B12 + B8A).
- **Sliding-Window Scene Mapping**: Automatically splits large satellite scenes ($H, W > 64$) into 64×64 patches and generates a 2D land-cover segmentation map and composition breakdown.
- **BDH Diagnostics**: Live measurement of post-ReLU activation sparsity (~92–93% zeros) across the 4 tied levels.

## Attention Complexity Clarification
- The default `attn_mode="scores"` computes $(QK^T)V$, forming a token-to-token attention matrix without softmax ($O(T^2 d)$ complexity).
- The state formulation computes $Q(K^T V)$ ($O(T d D)$ complexity).
- Under non-causal attention, both are mathematically equivalent, but because $T=64 < D=341$, the scores formulation is $\sim 5.3\times$ faster in practice for 64-patch vision inputs.

## Results
| Config | n_seeds | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | Params |
|---|---|---|---|---|---|---|
| bdh_main | 1 | 97.80 ± 0.00 | 97.72 ± 0.00 | 97.71 ± 0.00 | 97.71 ± 0.00 | 1,788,682 |
| cnn | 1 | 98.62 ± 0.00 | 98.58 ± 0.00 | 98.56 ± 0.00 | 98.56 ± 0.00 | 969,930 |
| vit | 1 | 96.89 ± 0.00 | 96.75 ± 0.00 | 96.72 ± 0.00 | 96.72 ± 0.00 | 1,812,234 |

## Limitations
- Not an exact BDH-GPU reproduction; no time-recurrent synaptic state across patches.
- Activation sparsity is measured, not assumed (see results/tables/*_sparsity.csv).
