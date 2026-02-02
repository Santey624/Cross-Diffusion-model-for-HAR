# ============================================================
# Sensor-Level Multimodal VAE
# 7 independent TemporalSingleModalVAE instances (each 3ch)
# ============================================================

import torch
import torch.nn as nn

from src.models.temporal_vae import TemporalSingleModalVAE


# Default specs: all sensors are 3ch, latent_dim=8, base_channels=32
SENSOR_SPECS = {
    "phone_acc":   {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_gyro":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_grav":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "phone_lacc":  {"in_channels": 3, "seq_len": 800, "latent_dim": 8, "base_channels": 32},
    "watch_acc":   {"in_channels": 3, "seq_len": 268, "latent_dim": 8, "base_channels": 32},
    "watch_gyro":  {"in_channels": 3, "seq_len": 268, "latent_dim": 8, "base_channels": 32},
    "glasses_acc": {"in_channels": 3, "seq_len": 80,  "latent_dim": 8, "base_channels": 32},
}

SENSOR_NAMES = list(SENSOR_SPECS.keys())


class SensorMultiModalVAE(nn.Module):
    """
    Multimodal VAE with one TemporalSingleModalVAE per sensor.

    Forward input:  dict {sensor_name: (B, T, 3)}
    Forward output: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
    """

    def __init__(self, sensor_specs=None):
        super().__init__()
        specs = sensor_specs or SENSOR_SPECS
        self.sensor_names = list(specs.keys())
        self.vaes = nn.ModuleDict({
            name: TemporalSingleModalVAE(**params)
            for name, params in specs.items()
        })

    def forward(self, sensor_data):
        """
        sensor_data: dict {sensor_name: Tensor (B, T, 3)}
        Returns: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
        """
        outputs = {}
        for name in self.sensor_names:
            if name in sensor_data:
                outputs[name] = self.vaes[name](sensor_data[name])
        return outputs
