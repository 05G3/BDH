import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class EuroSATMS(Dataset):
    def __init__(self, split, data_dir="data/processed",
                 stats_path="data/splits/norm_stats.json",
                 train=False, eps=1e-6, bands=None):
        """bands: None = all 13 bands, or a list of band indices (e.g. [3, 2, 1] = RGB)."""
        self.images = np.load(f"{data_dir}/{split}_images.npy")
        self.labels = np.load(f"{data_dir}/{split}_labels.npy")
        with open(stats_path) as f:
            stats = json.load(f)

        self.select = bands is not None
        self.band_idx = list(bands) if self.select else list(range(13))
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)[self.band_idx].view(-1, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)[self.band_idx].view(-1, 1, 1)
        self.train = train
        self.eps = eps

    def __len__(self):
        return len(self.labels)

    def _augment(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
        return x

    def __getitem__(self, i):
        img = self.images[i]
        if self.select:
            img = img[self.band_idx]
        x = torch.from_numpy(img)
        x = (x - self.mean) / (self.std + self.eps)
        if self.train:
            x = self._augment(x)
        return x, int(self.labels[i])
