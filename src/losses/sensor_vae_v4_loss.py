# ============================================================
# Loss for Sensor Shared+Private VAE (V4)
#
# Terms:
#   recon    — MSE reconstruction per sensor
#   kl_s     — KL(PoE posterior || N(0,1))  — shared latent
#   kl_p     — KL(private posterior || N(0,1)) per sensor
#   align    — MSE + cosine similarity between per-sensor mu_s predictions
#              Forces mu_s(phone_acc) ≈ mu_s(watch_acc) ≈ mu_s(glasses_acc)
#              so that PoE works for imputation from any sensor subset.
# ============================================================

import torch
import torch.nn.functional as F


# Same-type sensor groups — should share activity in z_shared
ALIGNMENT_GROUPS = [
    ["phone_acc", "watch_acc", "glasses_acc"],
    ["phone_gyro", "watch_gyro"],
]


def _kl(mu, logvar):
    """KL(N(mu,sigma²) || N(0,1)) — summed over (D,T), mean over B."""
    lv = logvar.clamp(-8.0, 8.0)
    return (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(dim=(1, 2)).mean()


def _alignment_loss(outputs, valid_sensors=None):
    """
    Align per-sensor mu_s predictions (before PoE) within same-type groups.
    MSE aligns magnitude, cosine similarity aligns direction.
    """
    loss  = 0.0
    count = 0

    for group in ALIGNMENT_GROUPS:
        present = [n for n in group if n in outputs]
        if valid_sensors is not None:
            present = [n for n in present if n in valid_sensors]
        if len(present) < 2:
            continue

        mus = [outputs[n]["mu_s"] for n in present]  # each (B, D, T)

        for i in range(len(mus)):
            for j in range(i + 1, len(mus)):
                mse = F.mse_loss(mus[i], mus[j])
                cos = 1.0 - F.cosine_similarity(
                    mus[i].flatten(1), mus[j].flatten(1), dim=1
                ).mean()
                loss  = loss + mse + 0.5 * cos
                count += 1

    return loss / max(count, 1)


def sensor_vae_v4_loss(outputs, batch, mu_shared, logvar_shared,
                        beta_shared=1e-3, beta_private=1e-3,
                        align_weight=0.0, valid_sensors=None):
    """
    Args:
        outputs:        dict {name: {"recon","mu_s","logvar_s","mu_p","logvar_p",...}}
        batch:          dict {name: (B, T, C)} — ground truth signals
        mu_shared:      PoE posterior mean   (B, D_shared, T_lat)
        logvar_shared:  PoE posterior logvar (B, D_shared, T_lat)
        beta_shared:    KL weight for shared latent
        beta_private:   KL weight for private latent
        align_weight:   weight for per-sensor mu_s alignment loss
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
        return zero, {"recon": zero, "kl_s": zero, "kl_p": zero, "align": zero, "total": zero}

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

    # Alignment on per-sensor mu_s (before PoE)
    align = _alignment_loss(outputs, valid_sensors)

    total = recon_loss + beta_shared * kl_s + beta_private * kl_p + align_weight * align

    parts = {
        "recon": recon_loss.detach(),
        "kl_s":  kl_s.detach(),
        "kl_p":  kl_p.detach(),
        "align": align.detach() if torch.is_tensor(align) else torch.tensor(align),
        "total": total.detach(),
    }
    return total, parts
