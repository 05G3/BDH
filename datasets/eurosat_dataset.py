import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class EuroSATMS(Dataset):
    def __init__(self, split, data_dir="data/processed",
                 stats_path="data/splits/norm_stats.json",
                 train=False, eps=1e-6):
        """
        split: "train", "val", or "test" — loads data/processed/{split}_images.npy
        """
        self.images = np.load(f"{data_dir}/{split}_images.npy")   # (N, 13, 64, 64) float32
        self.labels = np.load(f"{data_dir}/{split}_labels.npy")   # (N,)

        with open(stats_path) as f:
            stats = json.load(f)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(13, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(13, 1, 1)
        self.train = train
        self.eps = eps

    def __len__(self):
        return len(self.labels)

    def _augment(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])          # horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])          # vertical flip
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=[1, 2])   # 90° rotation
        return x

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx])          # (13, 64, 64), still raw values
        x = (x - self.mean) / (self.std + self.eps)      # normalize (train-set stats)
        if self.train:
            x = self._augment(x)
        return x, int(self.labels[idx])