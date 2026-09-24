import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.bdh import BDH, BDHConfig


class BDHClassifier(nn.Module):
    def __init__(self, in_channels=13, img_size=64, patch_size=8,
                 n_embd=256, n_layer=4, n_head=4, mlp_internal_dim_multiplier=8,
                 dropout=0.1, num_classes=10,
                 causal=False, use_rope=True, tie_weights=True):
        super().__init__()
        self.embed = PatchEmbedding(in_channels, patch_size, img_size, n_embd, use_rope)
        cfg = BDHConfig(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                        mlp_internal_dim_multiplier=mlp_internal_dim_multiplier,
                        dropout=dropout, causal=causal, use_rope=use_rope,
                        tie_weights=tie_weights)
        self.bdh = BDH(cfg)
        self.ln_f = nn.LayerNorm(n_embd, elementwise_affine=False, bias=False)
        self.head = nn.Linear(n_embd, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):          # (B, 13, 64, 64)
        x = self.embed(x)          # (B, 64, D)
        x = self.bdh(x)            # (B, 64, D)
        x = x.mean(dim=1)          # global average pooling -> (B, D)
        x = self.ln_f(x)
        return self.head(x)        # (B, num_classes)