# ============================================================
# Sensor Data Augmentation for IMU/Wearable Data
# Applied to raw sensor data before VAE encoding
# ============================================================

import torch
import numpy as np
import random
from typing import Dict, Optional


class SensorAugmentation:
    """
    Augmentation pipeline for 3-axis sensor data.
    All augmentations preserve the temporal structure while adding variety.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        scale_range: tuple = (0.8, 1.2),
        time_shift_ratio: float = 0.1,
        rotation_prob: float = 0.5,
        magnitude_warp_prob: float = 0.3,
        magnitude_warp_sigma: float = 0.2,
        enabled: bool = True,
    ):
        """
        Args:
            noise_std: Std of Gaussian noise (relative to signal std)
            scale_range: (min, max) for random amplitude scaling
            time_shift_ratio: Max ratio of sequence length to shift
            rotation_prob: Probability of applying random rotation
            magnitude_warp_prob: Probability of magnitude warping
            magnitude_warp_sigma: Std of magnitude warp curve
            enabled: If False, returns data unchanged
        """
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_ratio = time_shift_ratio
        self.rotation_prob = rotation_prob
        self.magnitude_warp_prob = magnitude_warp_prob
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.enabled = enabled

    def __call__(self, sensor_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a dict of sensor tensors.

        Args:
            sensor_data: Dict[sensor_name -> Tensor of shape (B, T, 3) or (T, 3)]

        Returns:
            Augmented sensor data with same structure
        """
        if not self.enabled:
            return sensor_data

        augmented = {}
        for name, data in sensor_data.items():
            augmented[name] = self._augment_single(data, name)

        return augmented

    def _augment_single(self, data: torch.Tensor, name: str) -> torch.Tensor:
        """Augment a single sensor tensor."""
        # Handle both batched (B, T, 3) and unbatched (T, 3)
        squeeze = False
        if data.dim() == 2:
            data = data.unsqueeze(0)
            squeeze = True

        B, T, C = data.shape
        device = data.device
        dtype = data.dtype

        # 1. Gaussian Noise
        if self.noise_std > 0:
            noise = torch.randn_like(data) * self.noise_std
            data = data + noise

        # 2. Random Scaling (per sample in batch)
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scales = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(
                self.scale_range[0], self.scale_range[1]
            )
            data = data * scales

        # 3. Time Shift (circular)
        if self.time_shift_ratio > 0:
            max_shift = int(T * self.time_shift_ratio)
            if max_shift > 0:
                shifts = torch.randint(-max_shift, max_shift + 1, (B,))
                for i in range(B):
                    if shifts[i] != 0:
                        data[i] = torch.roll(data[i], shifts[i].item(), dims=0)

        # 4. Random 3D Rotation (for accelerometer/gyroscope)
        if random.random() < self.rotation_prob and C == 3:
            # Apply same rotation to entire batch for consistency
            R = self._random_rotation_matrix(device, dtype)
            # data: (B, T, 3) -> (B*T, 3) @ R.T -> (B, T, 3)
            data = torch.einsum('btc,cd->btd', data, R)

        # 5. Magnitude Warping
        if random.random() < self.magnitude_warp_prob:
            # Generate smooth curve to multiply signal
            warp = self._generate_magnitude_warp(B, T, device, dtype)
            data = data * warp.unsqueeze(-1)

        if squeeze:
            data = data.squeeze(0)

        return data

    def _random_rotation_matrix(self, device, dtype) -> torch.Tensor:
        """Generate a random 3D rotation matrix."""
        # Random angles
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, 2 * np.pi)
        psi = random.uniform(0, 2 * np.pi)

        # Rotation matrices around each axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ], device=device, dtype=dtype)

        Ry = torch.tensor([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ], device=device, dtype=dtype)

        Rz = torch.tensor([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ], device=device, dtype=dtype)

        return Rz @ Ry @ Rx

    def _generate_magnitude_warp(self, B: int, T: int, device, dtype) -> torch.Tensor:
        """Generate smooth magnitude warping curves."""
        # Use low-frequency sinusoids
        num_knots = random.randint(2, 4)
        t = torch.linspace(0, 1, T, device=device, dtype=dtype)

        warp = torch.ones(B, T, device=device, dtype=dtype)
        for _ in range(num_knots):
            freq = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2 * np.pi)
            amp = random.gauss(0, self.magnitude_warp_sigma)
            warp = warp + amp * torch.sin(2 * np.pi * freq * t + phase)

        # Clamp to reasonable range
        warp = torch.clamp(warp, 0.5, 1.5)
        return warp


class MixupAugmentation:
    """
    Mixup augmentation in latent space.
    Linearly interpolates between pairs of samples.
    """

    def __init__(self, alpha: float = 0.2, enabled: bool = True):
        """
        Args:
            alpha: Beta distribution parameter (lower = less mixing)
            enabled: If False, returns data unchanged
        """
        self.alpha = alpha
        self.enabled = enabled

    def __call__(
        self,
        latents: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Apply mixup to latent tensors.

        Args:
            latents: Dict[sensor_name -> Tensor of shape (B, D, T)]
            labels: Optional class labels (B,) for soft label mixing

        Returns:
            mixed_latents, mixed_labels (or None if labels not provided)
        """
        if not self.enabled:
            return latents, labels

        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        B = next(iter(latents.values())).size(0)
        device = next(iter(latents.values())).device

        # Random permutation for mixing pairs
        perm = torch.randperm(B, device=device)

        # Mix all latents with same lambda and permutation
        mixed = {}
        for name, z in latents.items():
            mixed[name] = lam * z + (1 - lam) * z[perm]

        # Mix labels if provided (for soft labels)
        mixed_labels = None
        if labels is not None:
            # Return lambda for loss computation
            mixed_labels = (labels, labels[perm], lam)

        return mixed, mixed_labels


class CutMixAugmentation:
    """
    CutMix augmentation in latent space.
    Replaces temporal segments between samples.
    """

    def __init__(self, prob: float = 0.5, enabled: bool = True):
        """
        Args:
            prob: Probability of applying cutmix
            enabled: If False, returns data unchanged
        """
        self.prob = prob
        self.enabled = enabled

    def __call__(
        self,
        latents: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Apply cutmix to latent tensors.

        Args:
            latents: Dict[sensor_name -> Tensor of shape (B, D, T)]
            labels: Optional class labels

        Returns:
            mixed_latents, mixed_labels
        """
        if not self.enabled or random.random() > self.prob:
            return latents, labels

        B = next(iter(latents.values())).size(0)
        T = next(iter(latents.values())).size(2)
        device = next(iter(latents.values())).device

        # Random permutation
        perm = torch.randperm(B, device=device)

        # Random cut region (temporal)
        cut_ratio = random.uniform(0.2, 0.5)
        cut_len = int(T * cut_ratio)
        cut_start = random.randint(0, T - cut_len)

        # Cut and mix all latents
        mixed = {}
        for name, z in latents.items():
            z_mixed = z.clone()
            z_mixed[:, :, cut_start:cut_start + cut_len] = z[perm, :, cut_start:cut_start + cut_len]
            mixed[name] = z_mixed

        # Lambda is proportion of original sample
        lam = 1 - cut_ratio

        mixed_labels = None
        if labels is not None:
            mixed_labels = (labels, labels[perm], lam)

        return mixed, mixed_labels


def create_sensor_augmentation(
    mode: str = "default",
    enabled: bool = True
) -> SensorAugmentation:
    """
    Factory function to create augmentation pipeline.

    Args:
        mode: "default", "strong", "light", or "none"
        enabled: Override to disable all augmentations

    Returns:
        SensorAugmentation instance
    """
    configs = {
        "none": dict(
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            time_shift_ratio=0.0,
            rotation_prob=0.0,
            magnitude_warp_prob=0.0,
            enabled=False,
        ),
        "light": dict(
            noise_std=0.02,
            scale_range=(0.95, 1.05),
            time_shift_ratio=0.05,
            rotation_prob=0.2,
            magnitude_warp_prob=0.1,
            magnitude_warp_sigma=0.1,
            enabled=enabled,
        ),
        "default": dict(
            noise_std=0.05,
            scale_range=(0.8, 1.2),
            time_shift_ratio=0.1,
            rotation_prob=0.5,
            magnitude_warp_prob=0.3,
            magnitude_warp_sigma=0.2,
            enabled=enabled,
        ),
        "strong": dict(
            noise_std=0.1,
            scale_range=(0.6, 1.4),
            time_shift_ratio=0.2,
            rotation_prob=0.8,
            magnitude_warp_prob=0.5,
            magnitude_warp_sigma=0.3,
            enabled=enabled,
        ),
    }

    if mode not in configs:
        raise ValueError(f"Unknown augmentation mode: {mode}. Choose from {list(configs.keys())}")

    return SensorAugmentation(**configs[mode])
