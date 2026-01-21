from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class CogAgeVAEDataset(Dataset):
    """
    CogAge Dataset for VAE training (unsupervised)

    Devices:
      - Smartphone: Accelerometer, Gyroscope, Gravity, Linear Acc (12 ch)
      - Smartwatch: Accelerometer, Gyroscope (6 ch)
      - Glasses: Accelerometer ONLY (3 ch)  <-- JINS Gyro REMOVED
    """

    def __init__(self, root_dir, split="training", transform=None):
        """
        root_dir: data/cogage/python/arrays/{blho|bbh|state}
        split: "training" (Session #1) or "testing" (Session #2)
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        prefix = "train" if split == "training" else "test"

        # ---------- Smartphone (12 channels) ----------
        acc = np.load(self.root_dir / f"{prefix}Accelerometer.npy")
        gyro = np.load(self.root_dir / f"{prefix}Gyroscope.npy")
        grav = np.load(self.root_dir / f"{prefix}Gravity.npy")
        lin = np.load(self.root_dir / f"{prefix}LinearAcceleration.npy")

        self.phone = np.concatenate([acc, gyro, grav, lin], axis=-1)
        # shape: (N, 800, 12)

        # ---------- Smartwatch (6 channels) ----------
        ms_acc = np.load(self.root_dir / f"{prefix}MSAccelerometer.npy")
        ms_gyro = np.load(self.root_dir / f"{prefix}MSGyroscope.npy")

        self.watch = np.concatenate([ms_acc, ms_gyro], axis=-1)
        # shape: (N, 268, 6)

        # ---------- Glasses (3 channels, ACC ONLY) ----------
        jins_acc = np.load(self.root_dir / f"{prefix}JinsAccelerometer.npy")
        self.glasses = jins_acc
        # shape: (N, 80, 3)

        # ---------- Sanity checks ----------
        n = self.phone.shape[0]
        assert self.watch.shape[0] == n, "Watch sample count mismatch"
        assert self.glasses.shape[0] == n, "Glasses sample count mismatch"

        self.n_samples = n

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        phone = self.phone[idx].astype(np.float32)
        watch = self.watch[idx].astype(np.float32)
        glasses = self.glasses[idx].astype(np.float32)

        if self.transform is not None:
            phone, watch, glasses = self.transform(phone, watch, glasses)

        return {
            "phone": torch.from_numpy(phone),
            "watch": torch.from_numpy(watch),
            "glasses": torch.from_numpy(glasses),
        }
