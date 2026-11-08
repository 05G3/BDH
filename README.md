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
RoPE linear attention (no softmax, bidirectional), gating, low-rank decoder,
residual LayerNorm -> GAP -> Linear(256, 10).
Adapted from Pathway's BDH-GPU (arXiv:2509.26507, github.com/pathwaycom/bdh);
differences: non-causal attention, patch embedding and classification head instead of byte LM.

## Results
| Config | n_seeds | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | Params |
|---|---|---|---|---|---|---|
| bdh_main | 1 | 97.80 ± 0.00 | 97.72 ± 0.00 | 97.71 ± 0.00 | 97.71 ± 0.00 | 1,788,682 |
| cnn | 1 | 98.62 ± 0.00 | 98.58 ± 0.00 | 98.56 ± 0.00 | 98.56 ± 0.00 | 969,930 |
| vit | 1 | 96.89 ± 0.00 | 96.75 ± 0.00 | 96.72 ± 0.00 | 96.72 ± 0.00 | 1,812,234 |

## Limitations
- Not an exact BDH-GPU reproduction; no time-recurrent synaptic state across patches.
- Activation sparsity is measured, not assumed (see results/tables/*_sparsity.csv).
