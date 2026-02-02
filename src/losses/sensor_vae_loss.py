# ============================================================
# Sensor-Level VAE Loss
# Reconstruction + KL + Alignment for shared latent space
# ============================================================

import torch
import torch.nn.functional as F


# Sensor groups for alignment loss
ALIGNMENT_GROUPS = [
    ["phone_acc", "watch_acc", "glasses_acc"],   # accelerometers
    ["phone_gyro", "watch_gyro"],                 # gyroscopes
]


def alignment_loss(outputs):
    """
    Encourage same-type sensors (e.g., all accelerometers) to have
    similar latent representations (global-average-pooled mu).
    """
    loss = 0.0
    count = 0

    for group in ALIGNMENT_GROUPS:
        present = [name for name in group if name in outputs]
        if len(present) < 2:
            continue

        # Global average pool each mu: (B, D, T_SHARED) -> (B, D)
        mus = [outputs[name]["mu"].mean(dim=2) for name in present]

        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                loss = loss + F.mse_loss(mus[i], mus[j])
                count += 1

    return loss / max(count, 1)


def sensor_vae_loss(outputs, batch, beta=1e-4, align_weight=0.1):
    """
    outputs: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
    batch:   dict {sensor_name: Tensor (B, T, 3)}
    beta:    KL weight
    align_weight: alignment loss weight
    """
    keys = list(outputs.keys())

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

    align = alignment_loss(outputs)

    total_loss = recon_loss + beta * kl_loss + align_weight * align

    parts = {
        "recon": recon_loss.detach(),
        "kl": kl_loss.detach(),
        "align": align.detach() if torch.is_tensor(align) else torch.tensor(align),
        "total": total_loss.detach(),
    }
    return total_loss, parts
