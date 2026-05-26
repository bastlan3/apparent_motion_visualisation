"""
Experiment grid: compare curriculum schedules and noise types.

All experiments use:
  - shape_type="cube", motion_mode="both"  (single shape, rotation+translation)
  - N_TRAIN=200, N_EPOCHS=20, BATCH_SIZE=8
  - missing_frac=0.25 (25% explicit all-missing batches)

Curriculum variants (fixed gaussian noise):
  linear     – σ_max ramps linearly   0.15 → 1.5 over first 67% of epochs
  cosine     – cosine ease-in-out     0.15 → 1.5
  exp        – log-linear             0.15 → 1.5

Noise variants (fixed linear curriculum):
  gaussian   – x + σ·N(0,I)
  uniform    – x + σ·U(-√3, √3)  (same variance, bounded)
  saltpepper – pixels randomly set to 0/1 with prob ∝ σ

Best model (by trajectory PSNR) is saved as denoiser_best.pt.
"""

import math
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_denoise import CondUNet, reconstruct_trajectory
from motion_denoise.training import (
    train_epoch, val_epoch, compute_psnr, sigma_max_schedule,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN      = 200
N_VAL        = 40
N_TRAJ_EVAL  = 10
T            = 16
BATCH_SIZE   = 8
N_EPOCHS     = 20
LR           = 1e-3
GRAD_CLIP    = 1.0
MISSING_FRAC = 0.25
MISSING_SIGMA = 0.8
SIGMA_START  = 0.15
SIGMA_TARGET = 1.5
WARMUP_FRAC  = 0.67
BEST_CKPT    = "denoiser_best.pt"

VARIANTS = [
    # (label,              schedule,   noise_type)
    ("linear + gaussian",  "linear",   "gaussian"),    # baseline
    ("cosine + gaussian",  "cosine",   "gaussian"),
    ("exp + gaussian",     "exp",      "gaussian"),
    ("linear + uniform",   "linear",   "uniform"),
    ("linear + saltpepper","linear",   "saltpepper"),
]


class PrerenderedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def eval_trajectory_psnr(model, traj_data, n, device):
    model.eval()
    psnrs = []
    for sample in traj_data[:n]:
        targets = sample["targets"]
        recon   = reconstruct_trajectory(model, targets[0], targets[-1], T, device=device)
        psnrs.append(compute_psnr(recon[1:-1], targets[1:-1]))
    return sum(psnrs) / len(psnrs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Shape: cube | Motion: both (rotation+translation)")
    print(f"N_TRAIN={N_TRAIN}, N_EPOCHS={N_EPOCHS}, missing_frac={MISSING_FRAC}\n")

    # Pre-render once — all variants share the same data
    print("Pre-rendering training data (cube, both)...")
    ds_train = ApparentMotionDataset(
        n_sequences=N_TRAIN, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="mixed", seed=42,
    )
    train_data = [ds_train[i] for i in tqdm(range(N_TRAIN), desc="train")]

    print("Pre-rendering validation data...")
    ds_val = ApparentMotionDataset(
        n_sequences=N_VAL, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="mixed", seed=1000,
    )
    val_data = [ds_val[i] for i in tqdm(range(N_VAL), desc="val")]

    print("Pre-rendering all-missing trajectory data...")
    ds_traj = ApparentMotionDataset(
        n_sequences=N_TRAJ_EVAL, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="all_missing", seed=1000,
    )
    traj_data = [ds_traj[i] for i in tqdm(range(N_TRAJ_EVAL), desc="traj")]

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

    results = []

    for label, schedule, noise_type in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Variant: {label}")
        print(f"{'='*60}")

        model = CondUNet(base_ch=32, emb_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS, eta_min=1e-5
        )

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(1, N_EPOCHS + 1):
            smax = sigma_max_schedule(
                epoch, N_EPOCHS, SIGMA_START, SIGMA_TARGET,
                schedule=schedule, warmup_frac=WARMUP_FRAC,
            )
            tr_loss = train_epoch(
                model, train_loader, optimizer, device,
                sigma_max=smax, grad_clip=GRAD_CLIP,
                missing_frac=MISSING_FRAC, missing_sigma=MISSING_SIGMA,
                noise_type=noise_type,
            )
            val_loss, val_psnr = val_epoch(model, val_loader, device)
            scheduler.step()

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = " *"

            print(
                f"  ep {epoch:02d}/{N_EPOCHS} | σ_max={smax:.3f} | "
                f"tr={tr_loss:.4f} | val={val_loss:.4f} | PSNR={val_psnr:.1f} dB{marker}"
            )

        # Load best weights and evaluate trajectory PSNR
        model.load_state_dict(best_state)
        traj_psnr = eval_trajectory_psnr(model, traj_data, N_TRAJ_EVAL, device)
        print(f"  → trajectory PSNR (all-missing): {traj_psnr:.2f} dB")

        results.append((label, best_val_loss, traj_psnr, best_state))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Variant':<26} {'best val_loss':>14} {'traj PSNR':>10}")
    print("-"*60)
    best_traj_psnr = max(r[2] for r in results)
    for label, val_loss, traj_psnr, _ in sorted(results, key=lambda r: -r[2]):
        flag = " ← best" if traj_psnr == best_traj_psnr else ""
        print(f"  {label:<24} {val_loss:>14.4f} {traj_psnr:>9.2f} dB{flag}")
    print("="*60)

    # Save the best model (by trajectory PSNR)
    best = max(results, key=lambda r: r[2])
    print(f"\nSaving best model ({best[0]}) to {BEST_CKPT}")
    torch.save(
        {
            "label":      best[0],
            "schedule":   VARIANTS[[v[0] for v in VARIANTS].index(best[0])][1],
            "noise_type": VARIANTS[[v[0] for v in VARIANTS].index(best[0])][2],
            "val_loss":   best[1],
            "traj_psnr":  best[2],
            "model_state": best[3],
        },
        BEST_CKPT,
    )
    print("Done.")


if __name__ == "__main__":
    main()
