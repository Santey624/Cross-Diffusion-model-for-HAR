import torch
import torch.nn.functional as F


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(recon, modalities, mask):
    loss = 0.0
    for m, x in modalities.items():
        msk = mask[m].float().view(-1, 1, 1)
        loss += F.mse_loss(recon[m] * msk, x * msk)
    return loss
