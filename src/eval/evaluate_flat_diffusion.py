# ============================================================
# Evaluate Joint Flat Conditional Diffusion
# ============================================================

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.models.temporal_vae import TemporalMultiModalVAE
from src.models.flat_diffusion import (
    JointFlatDenoiser, make_schedule, ddim_sample,
    build_condition, MODALITY_IDS, FLAT_DIMS,
    TOTAL_COND_DIM, MAX_FLAT_DIM,
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

# Classifier-free guidance scale (1.0 = no guidance)
CFG_SCALE = 2.0

OUTPUT_DIR = Path("outputs/flat_diffusion_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Latent dims for reshaping back to temporal
LATENT_DIMS = {
    "phone":   (32, 100),
    "watch":   (32, 34),
    "glasses": (16, 10),
}


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Evaluating Joint Flat Diffusion - Missing: {MISSING_MODALITY.upper()}")
    print(f"DDIM Steps: {DDIM_STEPS}, CFG Scale: {CFG_SCALE}")
    print(f"{'='*60}\n")

    # Load config
    config = torch.load(DIFFUSION_DIR / "config.pt", map_location="cpu")
    T = config["T"]
    schedule_type = config["schedule"]
    stats = config["stats"]
    model_cfg = config["config"]

    print(f"Schedule: {schedule_type}, T: {T}")

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

    # Flatten
    phone_flat = phone_latents.reshape(N, -1)
    watch_flat = watch_latents.reshape(N, -1)
    glasses_flat = glasses_latents.reshape(N, -1)

    # Normalize
    phone_norm = (phone_flat - stats["phone"]["mean"]) / stats["phone"]["std"]
    watch_norm = (watch_flat - stats["watch"]["mean"]) / stats["watch"]["std"]
    glasses_norm = (glasses_flat - stats["glasses"]["mean"]) / stats["glasses"]["std"]

    norm_data = {"phone": phone_norm, "watch": watch_norm, "glasses": glasses_norm}

    # Load model
    ckpt_path = DIFFUSION_DIR / "best_model.pt"
    print(f"Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = JointFlatDenoiser(
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        dropout=0.0,  # No dropout at inference
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded from epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")

    # Build condition
    cond = build_condition(
        phone_flat=phone_norm if MISSING_MODALITY != "phone" else None,
        watch_flat=watch_norm if MISSING_MODALITY != "watch" else None,
        glasses_flat=glasses_norm if MISSING_MODALITY != "glasses" else None,
        target_modality=MISSING_MODALITY,
        device=DEVICE,
    )

    # Modality ID
    mod_id = torch.full((N,), MODALITY_IDS[MISSING_MODALITY],
                        device=DEVICE, dtype=torch.long)

    # Build schedule
    sched = make_schedule(T, schedule_type)

    # Sample
    print(f"\nSampling {N} latents with DDIM ({DDIM_STEPS} steps)...")

    if CFG_SCALE > 1.0:
        print(f"Using classifier-free guidance (scale={CFG_SCALE})")
        cond_zero = torch.zeros_like(cond)

        # CFG wrapper
        class CFGModel:
            def __init__(self, model, cond, cond_zero, mod_id, scale):
                self.model = model
                self.cond = cond
                self.cond_zero = cond_zero
                self.mod_id = mod_id
                self.scale = scale

            def __call__(self, z_t, t, _cond, _mod_id):
                eps_cond = self.model(z_t, t, self.cond, self.mod_id)
                eps_uncond = self.model(z_t, t, self.cond_zero, self.mod_id)
                return eps_uncond + self.scale * (eps_cond - eps_uncond)

        cfg_model = CFGModel(model, cond, cond_zero, mod_id, CFG_SCALE)
        imputed_norm = ddim_sample(
            model=cfg_model,
            cond=cond,
            modality_id=mod_id,
            target_modality=MISSING_MODALITY,
            sched=sched,
            T=T,
            ddim_steps=DDIM_STEPS,
            clip_range=5.0,
            device=DEVICE,
        )
    else:
        imputed_norm = ddim_sample(
            model=model,
            cond=cond,
            modality_id=mod_id,
            target_modality=MISSING_MODALITY,
            sched=sched,
            T=T,
            ddim_steps=DDIM_STEPS,
            clip_range=5.0,
            device=DEVICE,
        )

    # Denormalize
    target_mean = stats[MISSING_MODALITY]["mean"].to(DEVICE)
    target_std = stats[MISSING_MODALITY]["std"].to(DEVICE)
    imputed_flat = imputed_norm * target_std + target_mean

    # Ground truth (flat)
    gt_flat = {"phone": phone_flat, "watch": watch_flat, "glasses": glasses_flat}[MISSING_MODALITY].to(DEVICE)

    # Latent MSE (flat)
    latent_mse = F.mse_loss(imputed_flat, gt_flat).item()

    # Reshape to temporal: (N, D*T') → (N, D, T')
    D, T_seq = LATENT_DIMS[MISSING_MODALITY]
    imputed_temporal = imputed_flat.reshape(N, D, T_seq)
    gt_temporal = gt_flat.reshape(N, D, T_seq)

    # Decode with VAE
    print("Decoding with VAE...")
    with torch.no_grad():
        decoder = getattr(vae, MISSING_MODALITY).decoder
        imputed_signals = decoder(imputed_temporal)
        gt_signals = decoder(gt_temporal)

    imputed_signals = imputed_signals.cpu()
    gt_signals = gt_signals.cpu()

    signal_mse = F.mse_loss(imputed_signals, gt_signals).item()
    per_sample_mse = ((imputed_signals - gt_signals) ** 2).mean(dim=(1, 2))

    print(f"\n{'='*60}")
    print(f"RESULTS - Joint Flat Diffusion ({MISSING_MODALITY.upper()}):")
    print(f"  Latent MSE:     {latent_mse:.6f}")
    print(f"  Signal MSE:     {signal_mse:.6f}")
    print(f"  Per-sample MSE: {per_sample_mse.mean():.6f} ± {per_sample_mse.std():.6f}")
    print(f"{'='*60}")

    print(f"\nCOMPARISON:")
    print(f"  VAE Reconstruction:        0.006")
    print(f"  MLP Baseline (flat):       0.249")
    print(f"  Old 3D Diffusion (joint):  1.879")
    print(f"  New Flat Diffusion (joint): {latent_mse:.3f}")
    print(f"{'='*60}\n")

    # Visualize
    print("Creating visualizations...")
    num_plot = min(5, N)
    n_channels = imputed_signals.shape[2]

    for idx in range(num_plot):
        fig, axs = plt.subplots(min(n_channels, 6), 1,
                                figsize=(14, 3 * min(n_channels, 6)))
        if min(n_channels, 6) == 1:
            axs = [axs]

        for ch in range(min(n_channels, 6)):
            real = gt_signals[idx, :, ch].numpy()
            imputed = imputed_signals[idx, :, ch].numpy()

            axs[ch].plot(real, label="Ground Truth", alpha=0.8, linewidth=1.5)
            axs[ch].plot(imputed, label="Flat Diffusion", alpha=0.8, linewidth=1.5, linestyle='--')
            axs[ch].set_title(f"{MISSING_MODALITY.upper()} Ch{ch} "
                              f"(Sample MSE: {per_sample_mse[idx]:.4f})")
            axs[ch].legend()
            axs[ch].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{MISSING_MODALITY}_flat_sample_{idx}.png", dpi=150)
        plt.close()

    # Save metrics
    with open(OUTPUT_DIR / f"{MISSING_MODALITY}_flat_metrics.txt", "w") as f:
        f.write(f"Missing Modality: {MISSING_MODALITY}\n")
        f.write(f"Model: Joint Flat Conditional Diffusion\n")
        f.write(f"Schedule: {schedule_type}, T: {T}\n")
        f.write(f"DDIM Steps: {DDIM_STEPS}, CFG Scale: {CFG_SCALE}\n")
        f.write(f"Num Samples: {N}\n")
        f.write(f"Latent MSE: {latent_mse:.6f}\n")
        f.write(f"Signal MSE: {signal_mse:.6f}\n")

    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
