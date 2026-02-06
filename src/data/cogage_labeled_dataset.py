# ============================================================
# CogAge Sensor-Level Dataset with Activity Labels
# Returns 7 sensor modalities + activity label
# ============================================================

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


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


class CogAgeLabeledDataset(Dataset):
    """
    CogAge Dataset with activity labels.

    Returns dict with 7 sensor keys + "label" key.
    """

    def __init__(self, root_dir, split="training", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        prefix = "train" if split == "training" else "test"

        # Load sensor data
        self.data = {}
        for key, (suffix, _) in SENSOR_FILES.items():
            arr = np.load(self.root_dir / f"{prefix}{suffix}.npy")
            self.data[key] = arr

        # Load labels
        self.labels = np.load(self.root_dir / f"{prefix}Labels.npy")

        # Sanity check
        n = self.labels.shape[0]
        for k, v in self.data.items():
            assert v.shape[0] == n, f"{k} has {v.shape[0]} samples, expected {n}"

        self.n_samples = n

        # Build label mapping (some classes might be missing)
        unique_labels = np.unique(self.labels)
        self.label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: int(l) for i, l in enumerate(unique_labels)}
        self.n_classes = len(unique_labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {k: self.data[k][idx].astype(np.float32) for k in SENSOR_FILES}

        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to tensors
        out = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
               for k, v in sample.items()}

        # Add label (mapped to contiguous indices)
        raw_label = int(self.labels[idx])
        out["label"] = self.label_to_idx[raw_label]

        return out


def get_combined_labeled_dataset(data_roots, split, transform=None):
    """
    Combine multiple dataset roots (blho, bbh, state) with consistent label mapping.
    """
    from torch.utils.data import ConcatDataset

    datasets = []
    all_labels = []

    # First pass: collect all unique labels
    for root in data_roots.values():
        path = Path(root) / split
        prefix = "train" if split == "training" else "test"
        labels = np.load(path / f"{prefix}Labels.npy")
        all_labels.extend(labels.tolist())

    unique_labels = sorted(set(all_labels))
    label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
    n_classes = len(unique_labels)

    # Second pass: create datasets with shared mapping
    class CogAgeLabeledDatasetShared(Dataset):
        def __init__(self, root_dir, split, transform, label_map):
            self.root_dir = Path(root_dir) / split
            self.transform = transform
            self.label_to_idx = label_map

            prefix = "train" if split == "training" else "test"

            self.data = {}
            for key, (suffix, _) in SENSOR_FILES.items():
                self.data[key] = np.load(self.root_dir / f"{prefix}{suffix}.npy")

            self.labels = np.load(self.root_dir / f"{prefix}Labels.npy")
            self.n_samples = self.labels.shape[0]

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            sample = {k: self.data[k][idx].astype(np.float32) for k in SENSOR_FILES}
            if self.transform is not None:
                sample = self.transform(sample)
            out = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                   for k, v in sample.items()}
            out["label"] = self.label_to_idx[int(self.labels[idx])]
            return out

    for root in data_roots.values():
        ds = CogAgeLabeledDatasetShared(root, split, transform, label_to_idx)
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    combined.n_classes = n_classes
    combined.label_to_idx = label_to_idx
    combined.idx_to_label = {v: k for k, v in label_to_idx.items()}

    return combined
