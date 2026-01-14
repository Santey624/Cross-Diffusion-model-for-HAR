# src/data/dataset.py

import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

from src.data.modality_registry import MODALITIES


class CogAgeContractDataset(Dataset):
    def __init__(
        self,
        samples_path: str,
        indices: List[int],
        norm_path: Optional[str] = None,
        device_filter: Optional[List[str]] = None,  # e.g., ["phone"] for smartphone-only
    ):
        self.samples = torch.load(samples_path)
        self.indices = indices
        self.device_filter = device_filter

        self.norm = None
        if norm_path is not None:
            with open(norm_path, "r", encoding="utf-8") as f:
                self.norm = json.load(f)

    def __len__(self):
        return len(self.indices)

    def _apply_norm(self, m: str, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return x
        mean = torch.tensor(self.norm[m]["mean"], dtype=x.dtype)
        std = torch.tensor(self.norm[m]["std"], dtype=x.dtype)
        return (x - mean) / std

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[self.indices[idx]]

        modalities_out: Dict[str, torch.Tensor] = {}
        mask_out: Dict[str, int] = {}

        for m, spec in MODALITIES.items():
            # optionally filter by device (e.g., only phone)
            if self.device_filter is not None and spec.device not in self.device_filter:
                # force missing
                modalities_out[m] = torch.zeros_like(s["modalities"][m]).float()
                mask_out[m] = 0
                continue

            x = s["modalities"][m].float()
            mask_val = int(s["mask"][m])

            # normalize only if present
            if mask_val == 1:
                x = self._apply_norm(m, x)

            modalities_out[m] = x
            mask_out[m] = mask_val

        return {
            "id": s["id"],
            "subject": int(s["subject"]),
            "session": int(s["session"]),
            "label": int(s["label"]),
            "modalities": modalities_out,  # dict of [T,C]
            "mask": mask_out,              # dict of 0/1
        }
