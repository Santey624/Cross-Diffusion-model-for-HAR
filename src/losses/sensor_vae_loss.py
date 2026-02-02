# ============================================================
# Sensor-Level VAE Loss
# Reconstruction + KL for 7 sensor modalities
# ============================================================

import torch
import torch.nn.functional as F


def sensor_vae_loss(outputs, batch, beta=1e-4):
    """
    outputs: dict {sensor_name: {"recon", "mu", "logvar", "z"}}
    batch:   dict {sensor_name: Tensor (B, T, 3)}
    beta:    KL weight
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

    total_loss = recon_loss + beta * kl_loss

    parts = {
        "recon": recon_loss.detach(),
        "kl": kl_loss.detach(),
        "total": total_loss.detach(),
    }
    return total_loss, parts
