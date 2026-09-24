import torch
from torch.utils.data import DataLoader
from datasets.eurosat_dataset import EuroSATMS
from models.patch_embedding import patchify, PatchEmbedding

train_ds = EuroSATMS("train", train=True)

loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

x, y = next(iter(loader))

print("Batch:", x.shape, y.shape)
print("Per-band mean (should be ~0):",
      x.mean(dim=(0, 2, 3)).numpy().round(2))

print("Per-band std (should be ~1):",
      x.std(dim=(0, 2, 3)).numpy().round(2))

p = patchify(x, 8)
print("Patches:", p.shape)

# Verify patchify is correct
ref = x[0, :, :8, :8].permute(1, 2, 0).reshape(-1)

print("Patch 0 correct:",
      torch.allclose(p[0, 0], ref))

emb = PatchEmbedding(dim=256)

print("Embedded:", emb(x).shape)