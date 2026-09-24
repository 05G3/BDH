import torch
from models.classifier import BDHClassifier

configs = [
    dict(use_rope=True,  tie_weights=True,  causal=False),   # faithful default
    dict(use_rope=False, tie_weights=True,  causal=False),   # ablation: learned pos-embed
    dict(use_rope=True,  tie_weights=False, causal=False),   # ablation: untied weights
]
for cfg in configs:
    model = BDHClassifier(**cfg)
    x = torch.randn(32, 13, 64, 64)
    logits = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg}  logits={tuple(logits.shape)}  params={n_params:,}")
    assert logits.shape == (32, 10)
    assert not torch.isnan(logits).any()

# Overfit-one-batch check on the default (faithful) config
model = BDHClassifier()
x = torch.randn(16, 13, 64, 64)
y = torch.randint(0, 10, (16,))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)  # matches official train.py
loss_fn = torch.nn.CrossEntropyLoss()
for step in range(101):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")