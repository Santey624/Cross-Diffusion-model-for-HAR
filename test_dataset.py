from src.data.cogage_vae_dataset import CogAgeBLHOVAEDataset
from src.data.normalizer import MultiModalNormalizer
import torch

dataset = CogAgeBLHOVAEDataset(
    root_dir="data/cogage/python/arrays/blho",
    split="training"
)

print("Samples:", len(dataset))

sample = dataset[0]
print(sample["phone"].shape)    # (800, 12)
print(sample["watch"].shape)    # (267, 6)
print(sample["glasses"].shape)  # (80, 6)

normalizer = MultiModalNormalizer.load("data/blho_normalizer.npz")

# Dataset MIT Normalizer
dataset_norm = CogAgeBLHOVAEDataset(
    root_dir="data/cogage/python/arrays/blho",
    split="training",
    transform=normalizer
)

sample = dataset_norm[0]

print("Phone mean/std:",
      sample["phone"].mean().item(),
      sample["phone"].std().item())

print("Watch mean/std:",
      sample["watch"].mean().item(),
      sample["watch"].std().item())

print("Glasses mean/std:",
      sample["glasses"].mean().item(),
      sample["glasses"].std().item())


from torch.utils.data import DataLoader

loader = DataLoader(dataset_norm, batch_size=64, shuffle=False)

phone_all = []
for batch in loader:
    phone_all.append(batch["phone"])

phone_all = torch.cat(phone_all, dim=0)

print("Global phone mean:", phone_all.mean().item())
print("Global phone std:", phone_all.std().item())