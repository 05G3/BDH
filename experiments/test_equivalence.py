import os
import torch
from models.classifier import BDHClassifier

torch.manual_seed(0)

# --- 1) Backbone features: scores form vs state form, float64, random init ---
for kw in [dict(use_rope=True), dict(use_rope=False)]:
    a = BDHClassifier(attn_mode="scores", **kw).double().eval()
    b = BDHClassifier(attn_mode="state", **kw).double().eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(8, 13, 64, 64, dtype=torch.float64)
    with torch.no_grad():
        fa = a.bdh(a.embed(x))       # (8, 64, 256), LayerNorm'd so O(1) values
        fb = b.bdh(b.embed(x))
    diff = (fa - fb).abs().max().item()
    print(f"{kw}: max |scores - state| = {diff:.2e}")
    assert diff < 1e-8, "attention forms are NOT equivalent"
print("Equivalence test PASSED (float64)\n")

# --- 2) Cost model ---
N, D = 512, 256
scores_cost = lambda T: T * T * (N + D)
state_cost = lambda T: 2 * T * N * D
print(f"T=64:  scores {scores_cost(64):,}  state {state_cost(64):,}  "
      f"(state is {state_cost(64)/scores_cost(64):.1f}x costlier)")
print(f"Crossover T = 2ND/(N+D) = {2*N*D/(N+D):.0f} tokens\n")

# --- 3) Trained checkpoint (if present): predictions under both forms ---
ckpt_path = "results/checkpoints/bdh_main_best.pt"
if os.path.exists(ckpt_path) and os.path.exists("data/processed/test_images.npy"):
    from datasets.eurosat_dataset import EuroSATMS
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = {k: v for k, v in ckpt["config"].items() if k != "attn_mode"}
    a = BDHClassifier(attn_mode="scores", **cfg).eval()
    b = BDHClassifier(attn_mode="state", **cfg).eval()
    a.load_state_dict(ckpt["model_state"])
    b.load_state_dict(ckpt["model_state"])
    ds = EuroSATMS("test")
    x = torch.stack([ds[i][0] for i in range(256)])
    with torch.no_grad():
        la, lb = a(x), b(x)
    print("Trained model, 256 test images:")
    print(f"  max |logit diff| = {(la - lb).abs().max().item():.2e}")
    print(f"  identical predictions: {(la.argmax(1) == lb.argmax(1)).float().mean().item()*100:.1f}%")
else:
    print("Checkpoint or cached test arrays not found; skipped section 3.")
