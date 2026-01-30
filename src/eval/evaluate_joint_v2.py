# ============================================================
# Evaluate Joint Temporal Diffusion v2
# Improved DDIM: 200 steps, quadratic spacing, wider clipping
# ============================================================

from pathlib import Path
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.temporal_vae import TemporalMultiModalVAE
from src.models.joint_diffusion import create_joint_diffusion_model


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAE_CHECKPOINT = "checkpoints/vae_gpu_epoch_050.pt"
DIFFUSION_DIR = Path("checkpoints/joint_diffusion_v2")
LATENTS_DIR = Path("data/latents")

MISSING_MODALITY = "phone"
NUM_EVAL_SAMPLES = 100
DDIM_STEPS = 200        # Key fix #3: was 50

OUTPUT_DIR = Path("outputs/joint_v2_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COSINE SCHEDULE (must match training)
# ============================================================
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    return betas.float()


def make_schedule(T, schedule_type):
    if schedule_type == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"alpha_bar": alpha_bar}


# ============================================================
# IMPROVED DDIM SAMPLER
# ============================================================
@torch.no_grad()
def ddim_sample_v2(model, target_modality, shape, phone_cond, watch_cond,
                   glasses_cond, alpha_bar, T, ddim_steps=200):
    """
    Improved DDIM sampling:
    - Quadratic timestep spacing (fix #4)
    - Wider clipping range [-5, 5] (fix #5)
    - More steps (fix #3)
    """
    B = shape[0]
    device = next(model.parameters()).device
    z = torch.randn(shape, device=device)

    alpha_bar = alpha_bar.to(device)

    # Quadratic spacing: more steps near t=0 where details matter
    tau = torch.linspace(0, 1, ddim_steps + 1, device=device)
    tau = (tau ** 2 * (T - 1)).long()
    tau = tau.flip(0)  # T-1 → 0

    for i in tqdm(range(len(tau) - 1), desc="DDIM Sampling", leave=False):
        t_now = tau[i]
        t_next = tau[i + 1]

        t_batch = torch.full((B,), t_now, device=device, dtype=torch.long)

        noise_pred = model(
            target_modality=target_modality,
            z_t=z,
            t=t_batch,
            phone_latent=phone_cond,
            watch_latent=watch_cond,
            glasses_latent=glasses_cond,
        )

        ab_now = alpha_bar[t_now]
        ab_next = alpha_bar[t_next]

        # Predict x0
        pred_x0 = (z - torch.sqrt(1 - ab_now) * noise_pred) / torch.sqrt(ab_now)
        pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)  # Wider clipping

        # DDIM deterministic step
        dir_zt = torch.sqrt(1 - ab_next) * noise_pred
        z = torch.sqrt(ab_next) * pred_x0 + dir_zt

    return z


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print(f"Evaluating Joint Temporal Diffusion v2")
    print(f"Missing: {MISSING_MODALITY.upper()}, DDIM Steps: {DDIM_STEPS}")
    print(f"{'='*60}\n")

    # Load checkpoint
    ckpt = torch.load(DIFFUSION_DIR / "best_model.pt", map_location=DEVICE)
    T = ckpt["T"]
    schedule_type = ckpt["schedule"]
    cfg = ckpt["config"]
    print(f"Schedule: {schedule_type}, T: {T}, Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")

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

    # Load normalization stats
    stats = torch.load(DIFFUSION_DIR / "normalization_stats.pt", map_location=DEVICE)

    # Normalize
    phone_norm = (phone_latents.to(DEVICE) - stats['phone']['mean'].to(DEVICE)) / stats['phone']['std'].to(DEVICE)
    watch_norm = (watch_latents.to(DEVICE) - stats['watch']['mean'].to(DEVICE)) / stats['watch']['std'].to(DEVICE)
    glasses_norm = (glasses_latents.to(DEVICE) - stats['glasses']['mean'].to(DEVICE)) / stats['glasses']['std'].to(DEVICE)

    # Load model
    model = create_joint_diffusion_model(
        hidden_dim=cfg['hidden_dim'],
        num_heads=cfg['num_heads'],
        num_conv_blocks=cfg['num_conv_blocks'],
        num_attn_blocks=cfg['num_attn_blocks'],
        dropout=0.0,  # No dropout at inference
    ).to(DEVICE)

    # Load weights (handle dropout mismatch)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in ckpt['model_state'].items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.eval()

    # Determine target and conditions
    if MISSING_MODALITY == "phone":
        target_shape = phone_norm.shape
        phone_cond = None
        watch_cond = watch_norm
        glasses_cond = glasses_norm
        gt_latents = phone_latents
        target_stats = stats['phone']
    elif MISSING_MODALITY == "watch":
        target_shape = watch_norm.shape
        phone_cond = phone_norm
        watch_cond = None
        glasses_cond = glasses_norm
        gt_latents = watch_latents
        target_stats = stats['watch']
    elif MISSING_MODALITY == "glasses":
        target_shape = glasses_norm.shape
        phone_cond = phone_norm
        watch_cond = watch_norm
        glasses_cond = None
        gt_latents = glasses_latents
        target_stats = stats['glasses']

    # Build schedule
    sched = make_schedule(T, schedule_type)

    # Sample
    print(f"\nImputing {MISSING_MODALITY} latents...")
    imputed_norm = ddim_sample_v2(
        model=model,
        target_modality=MISSING_MODALITY,
        shape=target_shape,
        phone_cond=phone_cond,
        watch_cond=watch_cond,
        glasses_cond=glasses_cond,
        alpha_bar=sched["alpha_bar"],
        T=T,
        ddim_steps=DDIM_STEPS,
    )

    # Denormalize
    imputed = imputed_norm * target_stats['std'].to(DEVICE) + target_stats['mean'].to(DEVICE)

    # Latent MSE
    gt_device = gt_latents.to(DEVICE)
    latent_mse = F.mse_loss(imputed, gt_device).item()

    # Decode with VAE
    print("Decoding with VAE...")
    with torch.no_grad():
        decoder = getattr(vae, MISSING_MODALITY).decoder
        imputed_signals = decoder(imputed).cpu()
        gt_signals = decoder(gt_device).cpu()

    signal_mse = F.mse_loss(imputed_signals, gt_signals).item()
    per_sample = ((imputed_signals - gt_signals) ** 2).mean(dim=(1, 2))

    print(f"\n{'='*60}")
    print(f"RESULTS - Joint Temporal Diffusion v2 ({MISSING_MODALITY.upper()}):")
    print(f"  Latent MSE:     {latent_mse:.6f}")
    print(f"  Signal MSE:     {signal_mse:.6f}")
    print(f"  Per-sample MSE: {per_sample.mean():.6f} +/- {per_sample.std():.6f}")
    print(f"{'='*60}")
    print(f"\nCOMPARISON:")
    print(f"  VAE Reconstruction:       0.006")
    print(f"  MLP Baseline:             0.249")
    print(f"  Old 3D Joint (linear):    1.879")
    print(f"  v2 3D Joint (cosine+SNR): {latent_mse:.3f}")
    print(f"{'='*60}\n")

    # Visualize
    print("Creating visualizations...")
    n_ch = imputed_signals.shape[2]
    for idx in range(min(5, N)):
        fig, axs = plt.subplots(min(n_ch, 6), 1, figsize=(14, 3 * min(n_ch, 6)))
        if min(n_ch, 6) == 1:
            axs = [axs]
        for ch in range(min(n_ch, 6)):
            axs[ch].plot(gt_signals[idx, :, ch].numpy(), label="Ground Truth", alpha=0.8)
            axs[ch].plot(imputed_signals[idx, :, ch].numpy(), label="v2 Diffusion",
                         alpha=0.8, linestyle='--')
            axs[ch].set_title(f"{MISSING_MODALITY.upper()} Ch{ch}")
            axs[ch].legend()
            axs[ch].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{MISSING_MODALITY}_v2_sample_{idx}.png", dpi=150)
        plt.close()

    with open(OUTPUT_DIR / f"{MISSING_MODALITY}_v2_metrics.txt", "w") as f:
        f.write(f"Missing: {MISSING_MODALITY}\n")
        f.write(f"Schedule: {schedule_type}, T: {T}\n")
        f.write(f"DDIM Steps: {DDIM_STEPS}\n")
        f.write(f"Latent MSE: {latent_mse:.6f}\n")
        f.write(f"Signal MSE: {signal_mse:.6f}\n")

    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
