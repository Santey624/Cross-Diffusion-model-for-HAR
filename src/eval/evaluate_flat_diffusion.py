# ============================================================
# Evaluate Joint Flat Conditional Diffusion v2
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.temporal_vae import TemporalMultiModalVAE
from src.models.flat_diffusion import (
    JointFlatDenoiser, make_schedule, ddim_sample, FLAT_DIMS
)

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/vae_gpu_epoch_050.pt"
DIFFUSION_DIR = Path("checkpoints/flat_diffusion")
LATENTS_DIR = Path("data/latents")

MISSING_MODALITY = "phone"
NUM_EVAL_SAMPLES = 100
DDIM_STEPS = 200

OUTPUT_DIR = Path("outputs/flat_diffusion_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIMS = {
    "phone":   (32, 100),
    "watch":   (32, 34),
    "glasses": (16, 10),
}


def main():
    print(f"\n{'='*60}")
    print(f"Evaluating Flat Diffusion v2 - Missing: {MISSING_MODALITY.upper()}")
    print(f"DDIM Steps: {DDIM_STEPS}")
    print(f"{'='*60}\n")

    # Load config
    config = torch.load(DIFFUSION_DIR / "config.pt", map_location="cpu")
    T = config["T"]
    schedule_type = config["schedule"]
    stats = config["stats"]
    model_cfg = config["config"]

    # Load VAE
    print("Loading VAE...")
    vae = TemporalMultiModalVAE(z_phone=32, z_watch=32, z_glasses=16).to(DEVICE)
    vae_ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(vae_ckpt["model_state"])
    vae.eval()

    # Load latents
    print("Loading latents...")
    phone_latents = torch.load(LATENTS_DIR / "train_latents_phone_mu.pt")[:NUM_EVAL_SAMPLES]
    watch_latents = torch.load(LATENTS_DIR / "train_latents_watch_mu.pt")[:NUM_EVAL_SAMPLES]
    glasses_latents = torch.load(LATENTS_DIR / "train_latents_glasses_mu.pt")[:NUM_EVAL_SAMPLES]

    N = phone_latents.shape[0]

    # Flatten and normalize
    phone_flat = phone_latents.reshape(N, -1)
    watch_flat = watch_latents.reshape(N, -1)
    glasses_flat = glasses_latents.reshape(N, -1)

    phone_norm = ((phone_flat - stats["phone"]["mean"]) / stats["phone"]["std"]).to(DEVICE)
    watch_norm = ((watch_flat - stats["watch"]["mean"]) / stats["watch"]["std"]).to(DEVICE)
    glasses_norm = ((glasses_flat - stats["glasses"]["mean"]) / stats["glasses"]["std"]).to(DEVICE)

    # Load model
    ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    model = JointFlatDenoiser(
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded from epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")

    # Build condition kwargs (pass available modalities)
    cond_kwargs = {}
    if MISSING_MODALITY != "phone":
        cond_kwargs["phone_cond"] = phone_norm
    if MISSING_MODALITY != "watch":
        cond_kwargs["watch_cond"] = watch_norm
    if MISSING_MODALITY != "glasses":
        cond_kwargs["glasses_cond"] = glasses_norm

    # Build schedule
    sched = make_schedule(T, schedule_type)

    # Sample
    print(f"\nSampling {N} latents with DDIM ({DDIM_STEPS} steps)...")
    imputed_norm = ddim_sample(
        model=model,
        target_modality=MISSING_MODALITY,
        sched=sched,
        T=T,
        ddim_steps=DDIM_STEPS,
        clip_range=5.0,
        device=DEVICE,
        **cond_kwargs,
    )

    # Denormalize
    target_mean = stats[MISSING_MODALITY]["mean"].to(DEVICE)
    target_std = stats[MISSING_MODALITY]["std"].to(DEVICE)
    imputed_flat = imputed_norm * target_std + target_mean

    # Ground truth
    gt_flat = {"phone": phone_flat, "watch": watch_flat, "glasses": glasses_flat}[MISSING_MODALITY].to(DEVICE)

    latent_mse = F.mse_loss(imputed_flat, gt_flat).item()

    # Reshape to temporal and decode
    D, T_seq = LATENT_DIMS[MISSING_MODALITY]
    imputed_temporal = imputed_flat.reshape(N, D, T_seq)
    gt_temporal = gt_flat.reshape(N, D, T_seq)

    print("Decoding with VAE...")
    with torch.no_grad():
        decoder = getattr(vae, MISSING_MODALITY).decoder
        imputed_signals = decoder(imputed_temporal).cpu()
        gt_signals = decoder(gt_temporal).cpu()

    signal_mse = F.mse_loss(imputed_signals, gt_signals).item()
    per_sample_mse = ((imputed_signals - gt_signals) ** 2).mean(dim=(1, 2))

    print(f"\n{'='*60}")
    print(f"RESULTS - Flat Diffusion v2 ({MISSING_MODALITY.upper()}):")
    print(f"  Latent MSE:     {latent_mse:.6f}")
    print(f"  Signal MSE:     {signal_mse:.6f}")
    print(f"  Per-sample MSE: {per_sample_mse.mean():.6f} +/- {per_sample_mse.std():.6f}")
    print(f"{'='*60}")
    print(f"\nCOMPARISON:")
    print(f"  VAE Reconstruction:        0.006")
    print(f"  MLP Baseline (flat):       0.249")
    print(f"  Old 3D Diffusion (joint):  1.879")
    print(f"  Flat Diffusion v2 (joint): {latent_mse:.3f}")
    print(f"{'='*60}\n")

    # Visualize
    print("Creating visualizations...")
    num_plot = min(5, N)
    n_ch = imputed_signals.shape[2]
    for idx in range(num_plot):
        fig, axs = plt.subplots(min(n_ch, 6), 1, figsize=(14, 3 * min(n_ch, 6)))
        if min(n_ch, 6) == 1:
            axs = [axs]
        for ch in range(min(n_ch, 6)):
            axs[ch].plot(gt_signals[idx, :, ch].numpy(), label="Ground Truth", alpha=0.8)
            axs[ch].plot(imputed_signals[idx, :, ch].numpy(), label="Diffusion v2",
                         alpha=0.8, linestyle='--')
            axs[ch].set_title(f"{MISSING_MODALITY.upper()} Ch{ch}")
            axs[ch].legend()
            axs[ch].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{MISSING_MODALITY}_v2_sample_{idx}.png", dpi=150)
        plt.close()

    with open(OUTPUT_DIR / f"{MISSING_MODALITY}_v2_metrics.txt", "w") as f:
        f.write(f"Missing: {MISSING_MODALITY}\nLatent MSE: {latent_mse:.6f}\n"
                f"Signal MSE: {signal_mse:.6f}\n")

    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
