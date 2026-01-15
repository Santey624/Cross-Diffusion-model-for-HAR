import torch


def vae_loss(batch, output, beta=1e-3):
    """
    batch: dict with keys phone, watch, glasses
    output: dict from MultiModalVAE forward
    beta: KL weight
    """

    recon_phone = ((batch["phone"] - output["recon_phone"]) ** 2).mean()
    recon_watch = ((batch["watch"] - output["recon_watch"]) ** 2).mean()
    recon_glasses = ((batch["glasses"] - output["recon_glasses"]) ** 2).mean()

    recon_loss = recon_phone + recon_watch + recon_glasses

    kl = -0.5 * torch.mean(
        1 + output["logvar"]
        - output["mu"] ** 2
        - torch.exp(output["logvar"])
    )

    total_loss = recon_loss + beta * kl

    return total_loss, recon_loss, kl
