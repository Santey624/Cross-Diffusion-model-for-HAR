# ============================================================
# Loss for Sensor Shared+Private VAE (V4)
#
# Terms:
#   recon    — MSE reconstruction per sensor
#   kl_s     — KL(PoE posterior || N(0,1))  — shared latent
#   kl_p     — KL(private posterior || N(0,1)) per sensor
#
# No explicit alignment loss needed:
#   z_shared is structurally shared → R² ≈ 1.0 by design.
# ============================================================

import torch
import torch.nn.functional as F


def _kl(mu, logvar):
    """KL(N(mu,sigma²) || N(0,1)) — summed over (D,T), mean over B."""
    lv = logvar.clamp(-8.0, 8.0)
    return (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(dim=(1, 2)).mean()


def sensor_vae_v4_loss(outputs, batch, mu_shared, logvar_shared,
                        beta_shared=1e-3, beta_private=1e-3,
                        valid_sensors=None):
    """
    Args:
        outputs:        dict {name: {"recon","mu_s","logvar_s","mu_p","logvar_p",...}}
        batch:          dict {name: (B, T, C)} — ground truth signals
        mu_shared:      PoE posterior mean   (B, D_shared, T_lat)
        logvar_shared:  PoE posterior logvar (B, D_shared, T_lat)
        beta_shared:    KL weight for shared latent
        beta_private:   KL weight for private latent
        valid_sensors:  set of sensor names to include (None = all present)

    Returns:
        total_loss, parts_dict
    """
    keys = list(outputs.keys())
    if valid_sensors is not None:
        keys = [k for k in keys if k in valid_sensors]

    if len(keys) == 0:
        device = mu_shared.device
        zero   = torch.tensor(0.0, device=device)
        return zero, {"recon": zero, "kl_s": zero, "kl_p": zero, "total": zero}

    # Reconstruction
    recon_loss = sum(
        F.mse_loss(outputs[k]["recon"], batch[k], reduction="mean")
        for k in keys
    ) / len(keys)

    # Shared KL (single PoE posterior, not per-sensor)
    kl_s = _kl(mu_shared, logvar_shared)

    # Private KL (per sensor, averaged)
    kl_p = sum(
        _kl(outputs[k]["mu_p"], outputs[k]["logvar_p"])
        for k in keys
    ) / len(keys)

    total = recon_loss + beta_shared * kl_s + beta_private * kl_p

    parts = {
        "recon": recon_loss.detach(),
        "kl_s":  kl_s.detach(),
        "kl_p":  kl_p.detach(),
        "total": total.detach(),
    }
    return total, parts
