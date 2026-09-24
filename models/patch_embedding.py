import torch
import torch.nn as nn


def patchify(x: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, p*p*C)"""
    B, C, H, W = x.shape
    p = patch_size
    assert H % p == 0 and W % p == 0
    gh, gw = H // p, W // p
    x = x.reshape(B, C, gh, p, gw, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, gh * gw, p * p * C)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=13, patch_size=8, img_size=64, dim=256, use_rope=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_rope = use_rope
        num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, dim)
        if not use_rope:
            # Learned 2D-agnostic positional embedding — only needed when RoPE
            # (which is applied inside BDH's attention) is switched off.
            self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = patchify(x, self.patch_size)   # (B, 64, 832)
        x = self.proj(x)                   # (B, 64, D)
        if not self.use_rope:
            x = x + self.pos
        return x