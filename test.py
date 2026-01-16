from src.models.vae import MultiModalVAE
import torch

model = MultiModalVAE()
batch = {
    "phone": torch.randn(4, 800, 12),
    "watch": torch.randn(4, 268, 6),
    "glasses": torch.randn(4, 80, 3),
}

out = model(batch["phone"], batch["watch"], batch["glasses"])

for k, v in out.items():
    print(k, v.shape)
