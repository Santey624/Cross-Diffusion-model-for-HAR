from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.opportunity.opportunity_constants import (
    OPP_SENSOR_FILES, OPP_LABEL_TRACKS, DEFAULT_LABEL_TRACK, label_file_suffix,
)


class OpportunityLabeledDataset(Dataset):
    """
    Opportunity dataset at sensor level, with activity labels.

    Returns a dict with 14 sensor keys + "label".
    Used for downstream classification evaluation.

    Loads pre-windowed .npy arrays produced by
    src.data.opportunity.preprocess_opportunity.

    Opportunity has seven parallel label tracks (see OPP_LABEL_TRACKS).
    Pick which one to classify on via `label_track`; the default is
    locomotion.

    Directory layout (per split):
        root_dir/{training|testing}/
            {train|test}BackAcc.npy        -> (N, WINDOW, 3)
            {train|test}BackGyro.npy       -> (N, WINDOW, 3)
            ...
            {train|test}Labels_{track}.npy -> (N,)   one per track

    Raw class ids are non-contiguous (and 0 = Null); label_to_idx maps
    them to 0-based indices.
    """

    def __init__(
        self,
        root_dir,
        split: str = "training",
        transform: Optional[Callable] = None,
        label_track: str = DEFAULT_LABEL_TRACK,
    ):
        if label_track not in OPP_LABEL_TRACKS:
            raise ValueError(
                f"Unknown label_track {label_track!r}; "
                f"expected one of {list(OPP_LABEL_TRACKS)}"
            )
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.label_track = label_track

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        prefix = "train" if split == "training" else "test"

        self.data = {}
        for key, (suffix, _) in OPP_SENSOR_FILES.items():
            self.data[key] = np.load(self.root_dir / f"{prefix}{suffix}.npy")

        self.labels = np.load(
            self.root_dir / f"{prefix}{label_file_suffix(label_track)}.npy"
        )

        n = self.labels.shape[0]
        for k, v in self.data.items():
            assert v.shape[0] == n, f"{k} has {v.shape[0]} samples, expected {n}"
        self.n_samples = n

        unique_labels = np.unique(self.labels)
        self.label_to_idx = {int(l): i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: int(l) for i, l in enumerate(unique_labels)}
        self.n_classes = len(unique_labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {k: self.data[k][idx].astype(np.float32) for k in OPP_SENSOR_FILES}

        if self.transform is not None:
            sample = self.transform(sample)

        out = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
               for k, v in sample.items()}
        out["label"] = self.label_to_idx[int(self.labels[idx])]
        return out
