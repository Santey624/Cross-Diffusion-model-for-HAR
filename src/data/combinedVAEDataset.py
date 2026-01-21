from torch.utils.data import Dataset


class CombinedVAEDataset(Dataset):
    """
    Concatenates multiple CogAgeVAEDataset instances
    into a single dataset for unsupervised VAE training.
    """

    def __init__(self, datasets):
        """
        datasets: list of Dataset objects
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

        self.cum_lengths = []
        total = 0
        for l in self.lengths:
            total += l
            self.cum_lengths.append(total)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        for dataset, cum_len in zip(self.datasets, self.cum_lengths):
            if idx < cum_len:
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")
