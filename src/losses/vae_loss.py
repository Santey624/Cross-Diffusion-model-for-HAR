# src/losses/vae_loss.py

import torch.nn.functional as F
import torch

def vae_loss_single(x, recon, mu, logvar, beta=1e-3):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def vae_loss_multimodal(batch, outputs, beta=1e-3):
    losses = {}

    total = 0.0
    for key in ["phone", "watch", "glasses"]:
        loss, rec, kl = vae_loss_single(
            batch[key],
            outputs[key]["recon"],
            outputs[key]["mu"],
            outputs[key]["logvar"],
            beta
        )
        losses[key] = (loss, rec, kl)
        total += loss

    return total, losses

def temporal_vae_loss(x, recon, mu, logvar, beta=1e-3):
    recon_loss = F.mse_loss(recon, x)

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl, recon_loss, kl
