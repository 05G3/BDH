# Adapted from Pathway Technology's BDH-GPU (github.com/pathwaycom/bdh),
# arXiv:2509.26507 "The Dragon Hatchling". Original: causal, byte-level LM.
# Adaptation for image-patch classification:
#   - non-causal attention (patches have no sequence order to hide)
#   - no embedding table / LM head (patch embedding done outside; classifier
#     head added outside)
#   - RoPE and weight-tying preserved exactly as in the original

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 8   # N = mult * n_embd // n_head
    dropout: float = 0.1
    causal: bool = False        # False for images (no sequence order)
    use_rope: bool = True       # True = official RoPE; False = rely on external learned pos-embed
    tie_weights: bool = True    # True = official behaviour (one weight set, run n_layer times)


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    def __init__(self, n_head, n_embd, mlp_internal_dim_multiplier,
                 use_rope=True, causal=False):
        super().__init__()
        self.use_rope = use_rope
        self.causal = causal
        N = mlp_internal_dim_multiplier * n_embd // n_head
        if use_rope:
            self.register_buffer(
                "freqs", get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
            )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert K is Q
        _, _, T, _ = Q.size()
        if self.use_rope:
            r_phases = torch.arange(
                0, T, device=self.freqs.device, dtype=self.freqs.dtype
            ).view(1, 1, -1, 1) * self.freqs
            QR = self.rope(r_phases, Q)
            KR = QR
        else:
            QR, KR = Q, K
        scores = QR @ KR.mT
        if self.causal:
            scores = scores.tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    """BDH-GPU recurrent block, operating on an already-embedded sequence
    (B, T, D) instead of token ids. Run for config.n_layer steps."""

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D = config.n_head, config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        n_sets = 1 if config.tie_weights else config.n_layer

        self.encoder = nn.Parameter(torch.zeros(n_sets, nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(n_sets, nh, D, N).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros(n_sets, nh * N, D).normal_(std=0.02))

        self.attn = Attention(nh, D, config.mlp_internal_dim_multiplier,
                              config.use_rope, config.causal)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):            # x: (B, T, D)
        B, T, D = x.shape
        x = x.unsqueeze(1)           # (B, 1, T, D)
        x = self.ln(x)

        for level in range(self.config.n_layer):
            i = 0 if self.config.tie_weights else level
            encoder, encoder_v, decoder = self.encoder[i], self.encoder_v[i], self.decoder[i]

            x_latent = x @ encoder                       # (B, nh, T, N)
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)  # (B, nh, T, D)
            yKV = self.ln(yKV)

            y_latent = yKV @ encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            nh = self.config.n_head
            N = self.config.mlp_internal_dim_multiplier * D // nh
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        return x.view(B, T, D)