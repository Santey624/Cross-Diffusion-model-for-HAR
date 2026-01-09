import pickle


PATH = "C:/Users/shefq/Documents/Datasets/CogAge/python/dictionaries/left-hand-data/testing/subject1_Bending_event0_session2.pkl"

with open(PATH, "rb") as f:
    sample = pickle.load(f, encoding="latin1")

print("TYPE:", type(sample))
print("KEYS:", sample.keys())

print("\nDETAILS:")
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: {v}")