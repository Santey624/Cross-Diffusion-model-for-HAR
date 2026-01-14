# src/data/contract_builder.py

import torch
import numpy as np
from typing import Dict

from src.data.modality_registry import (
    MODALITIES,
    RAW_TO_CONTRACT,
    get_expected_length,
    get_num_channels,
)


# ============================================================
# Helper: fix sequence length (crop or ze
# ============================================================

def fix_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Crop or zero-pad a time series to target_length.
    Shape: [T, C]
    """
    T, C = x.shape

    if T > target_length:
        return x[:target_length]
    elif T < target_length:
        pad = torch.zeros(target_length - T, C, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    return x


# ============================================================
# Core: build one contract-compliant sample
# ============================================================

def build_contract_sample(raw_sample: Dict) -> Dict:
    """
    Convert a raw CogAge dictionary sample into a contract-compliant sample.
    """

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    subject = int(raw_sample["subject"])
    session = int(raw_sample["session"])
    label = int(raw_sample["label"])
    execution_id = int(raw_sample["executionId"])

    sample_id = f"S{subject}_sess{session}_exec{execution_id}"

    # --------------------------------------------------------
    # Initialize containers
    # --------------------------------------------------------
    modalities: Dict[str, torch.Tensor] = {}
    mask: Dict[str, int] = {}

    # --------------------------------------------------------
    # Fill all modalities defined in registry
    # --------------------------------------------------------
    for modality_name, spec in MODALITIES.items():

        T = spec.expected_length
        C = spec.channels

        # Default: modality missing
        x = torch.zeros(T, C, dtype=torch.float32)
        m = 0

        # Check if modality exists in raw sample
        for raw_key, contract_key in RAW_TO_CONTRACT.items():
            if contract_key == modality_name and raw_key in raw_sample:
                raw_x = raw_sample[raw_key]

                # Convert numpy → torch
                if isinstance(raw_x, np.ndarray):
                    raw_x = torch.from_numpy(raw_x).float()
                elif isinstance(raw_x, torch.Tensor):
                    raw_x = raw_x.float()
                else:
                    raise TypeError(
                        f"Unsupported data type for {raw_key}: {type(raw_x)}"
                    )

                # Ensure shape [T, C]
                if raw_x.ndim != 2:
                    raise ValueError(
                        f"{raw_key} has invalid shape {raw_x.shape}, expected 2D"
                    )

                # Fix length
                raw_x = fix_length(raw_x, T)

                x = raw_x
                m = 1
                break

        modalities[modality_name] = x
        mask[modality_name] = m

    # --------------------------------------------------------
    # Smartphone-only start: force mask for watch & jins
    # (can be removed later without refactor)
    # --------------------------------------------------------
    for modality_name, spec in MODALITIES.items():
        if spec.device in ["watch", "jins"]:
            mask[modality_name] = 0
            modalities[modality_name] = torch.zeros(
                spec.expected_length,
                spec.channels,
                dtype=torch.float32,
            )

    # --------------------------------------------------------
    # Final contract sample
    # --------------------------------------------------------
    contract_sample = {
        "id": sample_id,
        "subject": subject,
        "session": session,
        "task": "behavior_blho",
        "label": label,
        "execution_id": execution_id,
        "modalities": modalities,
        "mask": mask,
    }

    return contract_sample
