# ============================================================
# Sensor-Level Normalizer
# Per-sensor z-score normalization for 7 sensor modalities
# ============================================================

import numpy as np


class SensorNormalizer:
    """Z-score normalizer for sensor-level data (7 modalities, each 3ch)."""

    def __init__(self, stats):
        """
        stats: dict of {sensor_key: (mean, std)} where mean/std are (3,) arrays
        """
        self.stats = {k: (np.asarray(m, dtype=np.float32),
                          np.asarray(s, dtype=np.float32))
                      for k, (m, s) in stats.items()}

    def __call__(self, sample):
        """
        sample: dict of {sensor_key: np.ndarray (T, 3)}
        Returns: dict of normalized arrays
        """
        out = {}
        for k, v in sample.items():
            if k in self.stats:
                mean, std = self.stats[k]
                out[k] = (v - mean) / (std + 1e-8)
            else:
                out[k] = v
        return out

    def save(self, path):
        save_dict = {}
        for k, (m, s) in self.stats.items():
            save_dict[f"{k}_mean"] = m
            save_dict[f"{k}_std"] = s
        np.savez(path, **save_dict)

    @staticmethod
    def load(path):
        data = np.load(path)
        # Reconstruct stats dict from flat keys
        keys = set()
        for name in data.files:
            # e.g. "phone_acc_mean" -> "phone_acc"
            if name.endswith("_mean"):
                keys.add(name[:-5])
            elif name.endswith("_std"):
                keys.add(name[:-4])

        stats = {}
        for k in sorted(keys):
            stats[k] = (data[f"{k}_mean"], data[f"{k}_std"])
        return SensorNormalizer(stats)
