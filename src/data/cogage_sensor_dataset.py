# ============================================================
# CogAge Sensor-Level Dataset
# Returns 7 individual sensor modalities (each 3 channels)
# ============================================================

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable


# Sensor key -> (npy file suffix, sequence length)
SENSOR_FILES = {
    "phone_acc":   ("Accelerometer", 800),
    "phone_gyro":  ("Gyroscope", 800),
    "phone_grav":  ("Gravity", 800),
    "phone_lacc":  ("LinearAcceleration", 800),
    "watch_acc":   ("MSAccelerometer", 268),
    "watch_gyro":  ("MSGyroscope", 268),
    "glasses_acc": ("JinsAccelerometer", 80),
}

SENSOR_NAMES = list(SENSOR_FILES.keys())

DEVICE_GROUPS = {
    "phone": ["phone_acc", "phone_gyro", "phone_grav", "phone_lacc"],
    "watch": ["watch_acc", "watch_gyro"],
    "glasses": ["glasses_acc"],
}


class CogAgeSensorDataset(Dataset):
    """
    CogAge Dataset at sensor level.

    Returns a dict with 7 keys, each -> torch.Tensor of shape (T, 3).
    """

    def __init__(
        self,
        root_dir,
        split: str = "training",
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
    ):
        """
        root_dir: data/cogage/python/arrays/{blho|bbh|state}
        split: "training" or "testing"
        transform: Normalizer callable (applied after augmentation)
        augmentation: SensorAugmentation callable (applied before normalization)
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.augmentation = augmentation

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        prefix = "train" if split == "training" else "test"

        self.data = {}
        for key, (suffix, _) in SENSOR_FILES.items():
            arr = np.load(self.root_dir / f"{prefix}{suffix}.npy")
            self.data[key] = arr  # (N, T, 3)

        # Sanity check: all sensors have the same number of samples
        counts = {k: v.shape[0] for k, v in self.data.items()}
        n = list(counts.values())[0]
        for k, c in counts.items():
            assert c == n, f"{k} sample count {c} != {n}"

        self.n_samples = n

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {k: self.data[k][idx].astype(np.float32) for k in SENSOR_FILES}

        # Convert to tensors first for augmentation
        sample = {k: torch.from_numpy(v) for k, v in sample.items()}

        # Apply augmentation (if provided)
        if self.augmentation is not None:
            sample = self.augmentation(sample)

        # Apply normalization transform (if provided)
        if self.transform is not None:
            # Transform expects numpy, convert back
            sample_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                         for k, v in sample.items()}
            sample = self.transform(sample_np)
            sample = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                      for k, v in sample.items()}

        return sample
