import pickle
from src.data.contract_builder import build_contract_sample

PATH = "C:/Users/shefq/Documents/Datasets/CogAge/python/dictionaries/left-hand-data/testing/subject1_Bending_event0_session2.pkl"

with open(PATH, "rb") as f:
    raw = pickle.load(f, encoding="latin1")

sample = build_contract_sample(raw)

print(sample["id"])
for k, v in sample["modalities"].items():
    print(k, v.shape, "mask =", sample["mask"][k])
