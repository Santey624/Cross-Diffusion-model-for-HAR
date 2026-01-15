import torch
from torch.utils.data import DataLoader
import numpy as np

from src.data.cogage_vae_dataset import CogAgeBLHOVAEDataset
from src.data.normalizer import MultiModalNormalizer


dataset = CogAgeBLHOVAEDataset(
    root_dir="data/cogage/python/arrays/blho",
    split="training"
)

loader = DataLoader(dataset, batch_size=32, shuffle=False)

phone_all, watch_all, glasses_all = [], [], []

for batch in loader:
    phone_all.append(batch["phone"])
    watch_all.append(batch["watch"])
    glasses_all.append(batch["glasses"])

phone_all = torch.cat(phone_all, dim=0)     # (N, 800, 12)
watch_all = torch.cat(watch_all, dim=0)     # (N, 268, 6)
glasses_all = torch.cat(glasses_all, dim=0) # (N, 80, 6)

phone_mean = phone_all.mean(dim=(0, 1))
phone_std  = phone_all.std(dim=(0, 1))

watch_mean = watch_all.mean(dim=(0, 1))
watch_std  = watch_all.std(dim=(0, 1))

glasses_mean = glasses_all.mean(dim=(0, 1))
glasses_std  = glasses_all.std(dim=(0, 1))

normalizer = MultiModalNormalizer(
    phone_mean, phone_std,
    watch_mean, watch_std,
    glasses_mean, glasses_std
)

normalizer.save("data/blho_normalizer.npz")

print("Normalizer saved to data/blho_normalizer.npz")


