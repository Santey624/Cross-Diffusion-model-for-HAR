# src/data/modality_registry.py

from dataclasses import dataclass
from typing import Dict

# Global window definition
WINDOW_SEC: float = 5.0

# Modality specification
@dataclass(frozen=True)
class ModalitySpec:
    name: str
    sampling_rate: int
    channels: int
    device: str
    expected_length: int

# Modality registry (SINGLE SOURCE OF TRUTH)
MODALITIES: Dict[str, ModalitySpec] = {
    # Smartphone (Nexus 5X)
    "phone_acc": ModalitySpec(
        name="phone_acc",
        sampling_rate=200,
        channels=3,
        device="phone",
        expected_length=1000,  # 200 Hz * 5s
    ),
    "phone_gyro": ModalitySpec(
        name="phone_gyro",
        sampling_rate=200,
        channels=3,
        device="phone",
        expected_length=1000,
    ),
    "phone_grav": ModalitySpec(
        name="phone_grav",
        sampling_rate=200,
        channels=3,
        device="phone",
        expected_length=1000,
    ),
    "phone_lacc": ModalitySpec(
        name="phone_lacc",
        sampling_rate=200,
        channels=3,
        device="phone",
        expected_length=1000,
    ),
    "phone_mag": ModalitySpec(
        name="phone_mag",
        sampling_rate=50,
        channels=3,
        device="phone",
        expected_length=250,  # 50 Hz * 5s
    ),

    
    # Smartwatch (MS Band 2)
    "watch_acc": ModalitySpec(
        name="watch_acc",
        sampling_rate=67,
        channels=3,
        device="watch",
        expected_length=335,  # 67 Hz * 5s
    ),
    "watch_gyro": ModalitySpec(
        name="watch_gyro",
        sampling_rate=67,
        channels=3,
        device="watch",
        expected_length=335,
    ),

 
    # JINS MEME (cleaned)
    "jins_acc": ModalitySpec(
        name="jins_acc",
        sampling_rate=20,
        channels=3,
        device="jins",
        expected_length=100,  # 20 Hz * 5s
    ),
}



# Raw-key → Contract-modality mapping

RAW_TO_CONTRACT = {
    # Smartphone
    "Accelerometer": "phone_acc",
    "Gyroscope": "phone_gyro",
    "Gravity": "phone_grav",
    "LinearAccelerometer": "phone_lacc",
    "Magnetometer": "phone_mag",

    # Smartwatch
    "MSAccelerometer": "watch_acc",
    "MSGyroscope": "watch_gyro",

    # JINS MEME
    "JinsAccelerometer": "jins_acc",
}


# ============================================================
# Utility helpers
# ============================================================

def get_expected_length(modality_name: str) -> int:
    return MODALITIES[modality_name].expected_length


def get_num_channels(modality_name: str) -> int:
    return MODALITIES[modality_name].channels


def list_modalities():
    return list(MODALITIES.keys())
