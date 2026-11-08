import torch.nn as nn
from models.patch_embedding import PatchEmbedding


class SimpleCNN(nn.Module):
    """4 conv stages (3x3, BN, ReLU), 3 max-pools, GAP, linear head (~1M params)."""
    def __init__(self, in_channels=13, num_classes=10, widths=(64, 128, 256, 256), dropout=0.1):
        super().__init__()
        layers, c = [], in_channels
        for i, w in enumerate(widths):
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False),
                       nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))
        return self.head(self.drop(x))


class SimpleViT(nn.Module):
    """Same patch pipeline as BDH; standard pre-norm Transformer encoder with softmax
    attention and learned positional embeddings. depth=2 gives ~1.8M params, close to BDH."""
    def __init__(self, in_channels=13, img_size=64, patch_size=8, dim=256,
                 depth=2, heads=4, mlp_ratio=4, dropout=0.1, num_classes=10):
        super().__init__()
        self.embed = PatchEmbedding(in_channels, patch_size, img_size, dim, use_rope=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder(self.embed(x))
        return self.head(self.norm(x.mean(dim=1)))
