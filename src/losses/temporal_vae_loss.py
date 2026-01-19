import torch
import torch.nn.functional as F


def temporal_vae_loss(outputs, batch, beta=1e-4):
    keys = ["phone", "watch", "glasses"]

    recon_loss = 0.0
    kl_loss = 0.0

    for key in keys:
        x = batch[key]
        recon = outputs[key]["recon"]
        mu = outputs[key]["mu"]
        logvar = outputs[key]["logvar"]

        # Recon
        recon_loss = recon_loss + F.mse_loss(recon, x, reduction="mean")

        # Stabilize exp(logvar)
        logvar = torch.clamp(logvar, -8.0, 8.0)

        # KL: (B,D,T')
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum(dim=1)     # (B,T')
        kl = kl.mean(dim=1)    # (B,)
        kl = kl.mean()         # scalar

        kl_loss = kl_loss + kl

    # average over modalities
    recon_loss = recon_loss / len(keys)
    kl_loss = kl_loss / len(keys)

    total_loss = recon_loss + beta * kl_loss

    parts = {
        "recon": recon_loss.detach(),
        "kl": kl_loss.detach(),
        "total": total_loss.detach(),
    }
    return total_loss, parts
