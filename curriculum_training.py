"""
Curriculum training for trajectory reconstruction.

Two complementary ideas applied together:

1. sigma_max annealing
   σ_max starts at σ_start (≈ dataset noise_std) and linearly ramps to
   σ_target over `warmup_epochs`, then stays at σ_target.
   Early epochs: model only sees lightly corrupted frames → learns
   fine-detail denoising first.
   Late epochs: model sees σ up to 1.5+ → learns to reconstruct from
   context when the input is nearly pure noise (the all-missing regime).

2. Explicit all-missing batches (missing_frac)
   Each epoch, `missing_frac` of the batch items use a zeroed x_noisy
   + fixed σ=missing_sigma, exactly matching the reconstruct_trajectory
   inference scenario. This gives direct gradient signal for that task
   rather than relying purely on the large-σ proxy.
"""

import math
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_denoise import CondUNet, val_epoch, reconstruct_trajectory
from motion_denoise.training import train_epoch, compute_psnr

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN        = 500
N_VAL          = 50
T              = 16
BATCH_SIZE     = 8
N_EPOCHS       = 30
LR             = 1e-3
GRAD_CLIP      = 1.0
CHECKPOINT_PATH = "denoiser_curriculum.pt"

# Curriculum schedule
SIGMA_START    = 0.15   # matches dataset noise_std; model starts with easy task
SIGMA_TARGET   = 1.5    # full range by end of warmup
WARMUP_EPOCHS  = 20     # ramp over first 20 epochs; hold σ_target for remaining 10

# Explicit all-missing fraction
MISSING_FRAC   = 0.25   # 25% of each batch trained as fully-zeroed frames
MISSING_SIGMA  = 0.8    # σ value used for those items (same as inference)


def sigma_max_schedule(epoch: int) -> float:
    """Linear ramp from SIGMA_START to SIGMA_TARGET over WARMUP_EPOCHS."""
    t = min(epoch / WARMUP_EPOCHS, 1.0)
    return SIGMA_START + (SIGMA_TARGET - SIGMA_START) * t


class PrerenderedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nPre-rendering {N_TRAIN} training sequences...")
    train_data = [
        ApparentMotionDataset(n_sequences=N_TRAIN, T=T, corruption_mode="mixed", seed=42)[i]
        for i in tqdm(range(N_TRAIN), desc="render train")
    ]
    print(f"Pre-rendering {N_VAL} validation sequences...")
    val_data = [
        ApparentMotionDataset(n_sequences=N_VAL, T=T, corruption_mode="mixed", seed=1000)[i]
        for i in tqdm(range(N_VAL), desc="render val")
    ]
    print(f"Pre-rendering {N_VAL} all-missing sequences for trajectory demo...")
    traj_data = [
        ApparentMotionDataset(n_sequences=N_VAL, T=T, corruption_mode="all_missing", seed=1000)[i]
        for i in tqdm(range(N_VAL), desc="render traj")
    ]

    train_loader = DataLoader(
        PrerenderedDataset(train_data),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        PrerenderedDataset(val_data),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_collate_fn, num_workers=0,
    )

    model = CondUNet(base_ch=32, emb_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    print(f"\nCurriculum: σ_max {SIGMA_START}→{SIGMA_TARGET} over {WARMUP_EPOCHS} epochs, "
          f"then held. missing_frac={MISSING_FRAC}\n")

    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        sigma_max_curr = sigma_max_schedule(epoch)

        tr_loss = train_epoch(
            model, train_loader, optimizer, device,
            sigma_max=sigma_max_curr,
            grad_clip=GRAD_CLIP,
            missing_frac=MISSING_FRAC,
            missing_sigma=MISSING_SIGMA,
        )
        # Val always uses full sigma range for comparable metrics
        val_loss, val_psnr = val_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{N_EPOCHS} | σ_max={sigma_max_curr:.3f} | "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_PSNR={val_psnr:.2f} dB"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss":    val_loss,
                    "val_psnr":    val_psnr,
                },
                CHECKPOINT_PATH,
            )
            print(f"  → checkpoint saved (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss={best_val_loss:.4f} — checkpoint: {CHECKPOINT_PATH}")

    # ── Trajectory reconstruction demo ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Trajectory reconstruction: all intermediate frames missing")
    print("=" * 60)

    # Load best checkpoint for the demo
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    psnr_list = []
    for sample in traj_data[:10]:
        targets = sample["targets"]    # [T, 1, H, W]
        recon   = reconstruct_trajectory(model, targets[0], targets[-1], T, device=device)
        psnr_list.append(compute_psnr(recon[1:-1], targets[1:-1]))

    mean_psnr = sum(psnr_list) / len(psnr_list)
    print(f"\nPer-sequence PSNR (intermediate frames vs ground truth):")
    for i, p in enumerate(psnr_list):
        print(f"  sequence {i:02d}: {p:.2f} dB")
    print(f"\nMean PSNR over {len(psnr_list)} sequences: {mean_psnr:.2f} dB")
    print(
        f"\n(Baseline without curriculum: ~20 dB after 29 epochs)"
    )


if __name__ == "__main__":
    main()
