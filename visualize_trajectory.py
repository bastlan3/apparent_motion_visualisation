"""
Visualise a rotation+translation cube trajectory in three rows:
  1. Ground truth (all 16 frames)
  2. Input        (first + last clean; central 14 frames zeroed)
  3. Reconstruction (model output from endpoints only)

Loads the best model saved by experiment_grid.py.
Saves trajectory_visualization.png.
"""

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from motion_render import ApparentMotionDataset
from motion_denoise import CondUNet, reconstruct_trajectory
from motion_denoise.training import compute_psnr

BEST_CKPT  = "denoiser_best.pt"
OUT_FILE   = "trajectory_visualization.png"
T          = 16
VIS_SEED   = 7     # change to pick a different sequence


def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = CondUNet(base_ch=32, emb_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    label = ckpt.get("label", "unknown")
    print(f"Loaded model: {label}  (traj_PSNR={ckpt.get('traj_psnr', '?'):.2f} dB)")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(BEST_CKPT, device)

    # Generate a cube / both sequence
    ds = ApparentMotionDataset(
        n_sequences=1, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="all_missing", seed=VIS_SEED,
        trans_speed_range=(2.5, 4.0),
    )
    sample  = ds[0]
    targets = sample["targets"]   # [T, 1, H, W]  clean GT
    frames  = sample["frames"]    # [T, 1, H, W]  all-missing input
    mask    = sample["mask"]      # [T] bool

    # Reconstruct
    recon = reconstruct_trajectory(model, targets[0], targets[-1], T, device=device)

    # Per-frame PSNR (intermediate only)
    per_frame_psnr = [
        compute_psnr(recon[t], targets[t]) for t in range(1, T - 1)
    ]
    mean_psnr = sum(per_frame_psnr) / len(per_frame_psnr)
    print(f"Mean intermediate-frame PSNR: {mean_psnr:.2f} dB")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(T * 1.3, 5.5))
    gs  = GridSpec(
        4, T,
        figure=fig,
        hspace=0.08, wspace=0.04,
        height_ratios=[1, 1, 1, 0.18],
    )

    row_data = [
        ("Ground truth",              targets, None),
        ("Input (central missing)",   frames,  None),
        ("Reconstruction",            recon,   per_frame_psnr),
    ]

    for row_idx, (row_label, seq, psnrs) in enumerate(row_data):
        for t in range(T):
            ax  = fig.add_subplot(gs[row_idx, t])
            img = seq[t, 0].numpy().clip(0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])

            # Red border for missing frames in input row
            is_missing = (row_idx == 1) and (0 < t < T - 1)
            spine_color = "#cc3333" if is_missing else "#aaaaaa"
            spine_lw    = 1.8 if is_missing else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(spine_color)
                spine.set_linewidth(spine_lw)

            # Frame index label (bottom row only)
            if row_idx == 2:
                ax.set_xlabel(f"t={t}", fontsize=6, labelpad=2)

            # Per-frame PSNR below reconstruction frames (intermediate only)
            if row_idx == 2 and psnrs and 0 < t < T - 1:
                ax_psnr = fig.add_subplot(gs[3, t])
                ax_psnr.text(
                    0.5, 0.6, f"{psnrs[t-1]:.1f}",
                    ha="center", va="center",
                    fontsize=6.5, color="#226633",
                    transform=ax_psnr.transAxes,
                )
                ax_psnr.axis("off")
            elif row_idx == 2 and psnrs:
                ax_psnr = fig.add_subplot(gs[3, t])
                ax_psnr.axis("off")

        # Row label on the leftmost axis
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.set_ylabel(row_label, fontsize=8, labelpad=4)

    # Add mean PSNR annotation and coloured legend
    red_patch  = mpatches.Patch(color="#cc3333", label="missing frame (input)")
    fig.legend(
        handles=[red_patch], loc="lower left",
        fontsize=7, framealpha=0.85,
        bbox_to_anchor=(0.01, 0.01),
    )
    fig.suptitle(
        f"Cube — rotation + translation  |  reconstruction from endpoints only"
        f"  |  mean PSNR = {mean_psnr:.2f} dB",
        fontsize=10, y=0.99,
    )

    plt.savefig(OUT_FILE, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
