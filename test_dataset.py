import torch
from torch.utils.data import DataLoader

from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.normalizer import MultiModalNormalizer


DATASETS = {
    "BLHO": "data/cogage/python/arrays/blho",
    "BBH": "data/cogage/python/arrays/bbh",
    "STATE": "data/cogage/python/arrays/state",
}

NORMALIZER_PATH = "data/combined_normalizer.npz"
BATCH_SIZE = 32


def check_single_sample(ds, name):
    print(f"\n--- Checking single sample: {name} ---")
    sample = ds[0]

    for key in ["phone", "watch", "glasses"]:
        x = sample[key]
        print(
            f"{key:8s} shape={tuple(x.shape)}, "
            f"mean={x.mean():.4f}, std={x.std():.4f}"
        )

        assert torch.isfinite(x).all(), f"{key} contains NaN or Inf"


def check_dataloader(ds, name):
    print(f"\n--- Checking DataLoader: {name} ---")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    batch = next(iter(loader))
    for key in ["phone", "watch", "glasses"]:
        print(f"{key:8s} batch shape={tuple(batch[key].shape)}")

    return batch


def check_global_stats(ds, name):
    print(f"\n--- Checking GLOBAL stats: {name} ---")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    phone_all = []
    watch_all = []
    glasses_all = []

    for batch in loader:
        phone_all.append(batch["phone"])
        watch_all.append(batch["watch"])
        glasses_all.append(batch["glasses"])

    phone_all = torch.cat(phone_all, dim=0)
    watch_all = torch.cat(watch_all, dim=0)
    glasses_all = torch.cat(glasses_all, dim=0)

    print("PHONE   global mean/std:",
          phone_all.mean().item(),
          phone_all.std().item())

    print("WATCH   global mean/std:",
          watch_all.mean().item(),
          watch_all.std().item())

    print("GLASSES global mean/std:",
          glasses_all.mean().item(),
          glasses_all.std().item())


def main():
    print("Loading normalizer...")
    normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

    for name, root in DATASETS.items():
        print(f"\n==============================")
        print(f"DATASET: {name}")
        print(f"==============================")

        ds = CogAgeVAEDataset(
            root_dir=root,
            split="training",
            transform=normalizer
        )

        print(f"Number of samples: {len(ds)}")

        check_single_sample(ds, name)
        batch = check_dataloader(ds, name)
        check_global_stats(ds, name)

    print("\n✅ ALL DATASET TESTS PASSED")


if __name__ == "__main__":
    main()
