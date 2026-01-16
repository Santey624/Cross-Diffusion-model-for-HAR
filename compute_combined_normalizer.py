import torch
from torch.utils.data import DataLoader

from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer


DATASETS = [
    "data/cogage/python/arrays/blho",
    "data/cogage/python/arrays/bbh",
    "data/cogage/python/arrays/state",
]

BATCH_SIZE = 32


def collect_stats():
    phone_all, watch_all, glasses_all = [], [], []

    for root in DATASETS:
        ds = CogAgeVAEDataset(
            root_dir=root,
            split="training",
            transform=None
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        for batch in loader:
            phone_all.append(batch["phone"])
            watch_all.append(batch["watch"])
            glasses_all.append(batch["glasses"])

    phone_all = torch.cat(phone_all, dim=0)
    watch_all = torch.cat(watch_all, dim=0)
    glasses_all = torch.cat(glasses_all, dim=0)

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

    normalizer.save("data/combined_normalizer.npz")
    print("Saved combined normalizer to data/combined_normalizer.npz")


if __name__ == "__main__":
    collect_stats()
