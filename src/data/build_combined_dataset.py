from src.data.cogage_vae_dataset import CogAgeVAEDataset
from src.data.combinedVAEDataset import CombinedVAEDataset
from src.data.normalizer import MultiModalNormalizer


NORMALIZER_PATH = "data/combined_normalizer.npz"

normalizer = MultiModalNormalizer.load(NORMALIZER_PATH)

# -------- TRAINING (Session #1) --------
ds_blho_train = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/blho",
    split="training",
    transform=normalizer
)

ds_bbh_train = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/bbh",
    split="training",
    transform=normalizer
)

ds_state_train = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/state",
    split="training",
    transform=normalizer
)

train_dataset = CombinedVAEDataset([
    ds_blho_train,
    ds_bbh_train,
    ds_state_train
])

# -------- TESTING (Session #2) --------
ds_blho_test = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/blho",
    split="testing",
    transform=normalizer
)

ds_bbh_test = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/bbh",
    split="testing",
    transform=normalizer
)

ds_state_test = CogAgeVAEDataset(
    root_dir="data/cogage/python/arrays/state",
    split="testing",
    transform=normalizer
)

test_dataset = CombinedVAEDataset([
    ds_blho_test,
    ds_bbh_test,
    ds_state_test
])

print("Train samples:", len(train_dataset))
print("Test samples :", len(test_dataset))

sample = train_dataset[0]
print(sample["phone"].shape)
print(sample["watch"].shape)
print(sample["glasses"].shape)