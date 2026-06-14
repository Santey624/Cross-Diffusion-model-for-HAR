from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent

data_root = project_root / "dataset" / "OpportunityUCIDataset"

def get_dat_files():
    files_data = list(data_root.rglob("S1-ADL1.dat"))
    return files_data

def read_data():
    dat_file = data_root / "dataset" / "S1-ADL2.dat"
    df = pd.read_csv(dat_file, sep=r'\s+', header=None)
    print(f"Data shape: {df.shape}")
    print(df.head())
    return df

read_data()
print("Data read successfully.")