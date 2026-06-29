from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.opportunity.opportunity_constants import OPP_SENSOR_FILES


class OpportunitySensorDataset(Dataset):
    """
    Opportunity dataset at sensor level — no labels.

    Returns a dict with 14 keys, each -> torch.Tensor of shape (WINDOW, 3).
    Used for unsupervised diffusion model training.

    Loads pre-windowed .npy arrays produced by
    src.data.opportunity.preprocess_opportunity.

    Directory layout (per split):
        root_dir/{training|testing}/
            {train|test}BackAcc.npy   -> (N, WINDOW, 3)
            {train|test}BackGyro.npy  -> (N, WINDOW, 3)
            ...
    """

    def __init__(
        self,
        root_dir,
        split: str = "training",
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.augmentation = augmentation

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        prefix = "train" if split == "training" else "test"

        self.data = {}
        for key, (suffix, _) in OPP_SENSOR_FILES.items():
            self.data[key] = np.load(self.root_dir / f"{prefix}{suffix}.npy")

        counts = {k: v.shape[0] for k, v in self.data.items()}
        n = list(counts.values())[0]
        for k, c in counts.items():
            assert c == n, f"{k} sample count {c} != {n}"
        self.n_samples = n

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {k: self.data[k][idx].astype(np.float32) for k in OPP_SENSOR_FILES}
        sample = {k: torch.from_numpy(v) for k, v in sample.items()}

        if self.augmentation is not None:
            sample = self.augmentation(sample)

        if self.transform is not None:
            sample_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                         for k, v in sample.items()}
            sample = self.transform(sample_np)
            sample = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                      for k, v in sample.items()}

        return sample
