# ============================================================
# Sensor-Level VAE Loss with Sensor Masking
# Supports partial sensor data (e.g., WISDM has only 4/7 sensors)
# ============================================================

import torch
import torch.nn.functional as F


ALIGNMENT_GROUPS = [
    ["phone_acc", "watch_acc", "glasses_acc"],   # accelerometers
    ["phone_gyro", "watch_gyro"],                 # gyroscopes
]


def alignment_loss(outputs, valid_sensors=None):
    """Alignment loss, only between sensors that have real data."""
    loss = 0.0
    count = 0

    for group in ALIGNMENT_GROUPS:
        present = [name for name in group if name in outputs]
        if valid_sensors is not None:
            present = [name for name in present if name in valid_sensors]
        if len(present) < 2:
            continue

        mus = [outputs[name]["mu"].mean(dim=2) for name in present]

        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                loss = loss + F.mse_loss(mus[i], mus[j])
                count += 1

    return loss / max(count, 1)


def sensor_vae_loss_masked(outputs, batch, beta=1e-4, align_weight=0.1,
                           valid_sensors=None):
    """
    VAE loss that only computes on valid sensors.

    Args:
        outputs: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
        batch: dict {sensor_name: Tensor (B, T, 3)}
        beta: KL weight
        align_weight: alignment loss weight
        valid_sensors: set of sensor names to compute loss on.
                       If None, uses all sensors.
    """
    keys = list(outputs.keys())
    if valid_sensors is not None:
        keys = [k for k in keys if k in valid_sensors]

    if len(keys) == 0:
        zero = torch.tensor(0.0, device=next(iter(outputs.values()))["mu"].device)
        return zero, {"recon": zero, "kl": zero, "align": zero, "total": zero}

    recon_loss = 0.0
    kl_loss = 0.0

    for key in keys:
        x = batch[key]
        recon = outputs[key]["recon"]
        mu = outputs[key]["mu"]
        logvar = outputs[key]["logvar"]

        recon_loss = recon_loss + F.mse_loss(recon, x, reduction="mean")

        logvar = torch.clamp(logvar, -8.0, 8.0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum(dim=1).mean(dim=1).mean()
        kl_loss = kl_loss + kl

    recon_loss = recon_loss / len(keys)
    kl_loss = kl_loss / len(keys)

    align = alignment_loss(outputs, valid_sensors)

    total_loss = recon_loss + beta * kl_loss + align_weight * align

    parts = {
        "recon": recon_loss.detach(),
        "kl": kl_loss.detach(),
        "align": align.detach() if torch.is_tensor(align) else torch.tensor(align),
        "total": total_loss.detach(),
    }
    return total_loss, parts
