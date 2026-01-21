import numpy as np


class MultiModalNormalizer:
    def __init__(self, phone_mean, phone_std,
                       watch_mean, watch_std,
                       glasses_mean, glasses_std):

        # alles als NumPy speichern
        self.phone_mean = np.asarray(phone_mean)
        self.phone_std = np.asarray(phone_std)

        self.watch_mean = np.asarray(watch_mean)
        self.watch_std = np.asarray(watch_std)

        self.glasses_mean = np.asarray(glasses_mean)
        self.glasses_std = np.asarray(glasses_std)

    def __call__(self, phone, watch, glasses):
        phone = (phone - self.phone_mean) / (self.phone_std + 1e-8)
        watch = (watch - self.watch_mean) / (self.watch_std + 1e-8)
        glasses = (glasses - self.glasses_mean) / (self.glasses_std + 1e-8)
        return phone, watch, glasses

    def save(self, path):
        np.savez(
            path,
            phone_mean=self.phone_mean,
            phone_std=self.phone_std,
            watch_mean=self.watch_mean,
            watch_std=self.watch_std,
            glasses_mean=self.glasses_mean,
            glasses_std=self.glasses_std,
        )

    @staticmethod
    def load(path):
        data = np.load(path)
        return MultiModalNormalizer(
            data["phone_mean"], data["phone_std"],
            data["watch_mean"], data["watch_std"],
            data["glasses_mean"], data["glasses_std"],
        )
