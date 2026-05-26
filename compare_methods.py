"""
Side-by-side comparison of DSM denoiser vs Flow Matching reconstructions.

4 rows × T columns:
  1. Ground truth
  2. Input (central frames missing)
  3. DSM reconstruction   (from denoiser_best.pt)
  4. Flow field           (from flow_best.pt, Heun ODE integration)

Per-frame PSNR shown below rows 3 and 4.
Saves: compare_methods.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from motion_render import ApparentMotionDataset
from motion_denoise import CondUNet, reconstruct_trajectory
from motion_denoise.training import compute_psnr as dsm_compute_psnr
from motion_flow import FlowNet, reconstruct_flow_trajectory, compute_psnr as flow_compute_psnr

DSM_CKPT   = "denoiser_best.pt"
FLOW_CKPT  = "flow_best.pt"
OUT_FILE   = "compare_methods.png"
T          = 16
VIS_SEED   = 7


def load_dsm(path, device):
    ckpt  = torch.load(path, map_location=device)
    model = CondUNet(base_ch=32, emb_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    label = ckpt.get("label", "DSM")
    psnr  = ckpt.get("traj_psnr", float("nan"))
    print(f"DSM model: {label}  (traj_PSNR={psnr:.2f} dB)")
    return model


def load_flow(path, device):
    ckpt  = torch.load(path, map_location=device)
    model = FlowNet(base_ch=32, emb_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    psnr  = ckpt.get("traj_psnr", float("nan"))
    print(f"Flow model: traj_PSNR={psnr:.2f} dB")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dsm_model  = load_dsm(DSM_CKPT,  device)
    flow_model = load_flow(FLOW_CKPT, device)

    ds = ApparentMotionDataset(
        n_sequences=1, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="all_missing", seed=VIS_SEED,
        trans_speed_range=(2.5, 4.0),
    )
    sample  = ds[0]
    targets = sample["targets"]
    frames  = sample["frames"]

    dsm_recon  = reconstruct_trajectory(dsm_model,  targets[0], targets[-1], T, device=device)
    flow_recon = reconstruct_flow_trajectory(flow_model, targets[0], targets[-1], T, device=device)

    def per_frame_psnr(recon):
        return [dsm_compute_psnr(recon[t], targets[t]) for t in range(1, T - 1)]

    dsm_psnrs  = per_frame_psnr(dsm_recon)
    flow_psnrs = per_frame_psnr(flow_recon)

    dsm_mean  = sum(dsm_psnrs)  / len(dsm_psnrs)
    flow_mean = sum(flow_psnrs) / len(flow_psnrs)
    print(f"DSM  mean intermediate PSNR: {dsm_mean:.2f} dB")
    print(f"Flow mean intermediate PSNR: {flow_mean:.2f} dB")

    # ── Figure ────────────────────────────────────────────────────────────────
    N_ROWS = 6   # GT / input / DSM / DSM PSNR / Flow / Flow PSNR
    fig = plt.figure(figsize=(T * 1.3, 8.0))
    gs  = GridSpec(
        N_ROWS, T, figure=fig,
        hspace=0.06, wspace=0.04,
        height_ratios=[1, 1, 1, 0.18, 1, 0.18],
    )

    rows = [
        (0, "Ground truth",                  targets, None),
        (1, "Input (central missing)",        frames,  None),
        (2, f"DSM  ({dsm_mean:.1f} dB)",     dsm_recon,  dsm_psnrs),
        (4, f"Flow ({flow_mean:.1f} dB)",    flow_recon, flow_psnrs),
    ]
    psnr_rows = {2: 3, 4: 5}   # content_row → psnr_row in GridSpec

    for gs_row, row_label, seq, psnrs in rows:
        for t in range(T):
            ax  = fig.add_subplot(gs[gs_row, t])
            img = seq[t, 0].numpy().clip(0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])

            is_missing = (gs_row == 1) and (0 < t < T - 1)
            ec = "#cc3333" if is_missing else "#aaaaaa"
            lw = 1.8 if is_missing else 0.5
            for sp in ax.spines.values():
                sp.set_edgecolor(ec); sp.set_linewidth(lw)

            if gs_row == 4:
                ax.set_xlabel(f"t={t}", fontsize=6, labelpad=2)

        # Row label on leftmost cell
        ax0 = fig.add_subplot(gs[gs_row, 0])
        ax0.set_ylabel(row_label, fontsize=7.5, labelpad=4)

        # Per-frame PSNR strip (intermediate frames only)
        if psnrs is not None:
            pr = psnr_rows[gs_row]
            color = "#226633" if gs_row == 2 else "#22446b"
            for t in range(T):
                ax_p = fig.add_subplot(gs[pr, t])
                ax_p.axis("off")
                if 0 < t < T - 1:
                    ax_p.text(
                        0.5, 0.6, f"{psnrs[t-1]:.1f}",
                        ha="center", va="center", fontsize=6,
                        color=color, transform=ax_p.transAxes,
                    )

    red_patch = mpatches.Patch(color="#cc3333", label="missing frame (input)")
    fig.legend(handles=[red_patch], loc="lower left", fontsize=7,
               framealpha=0.85, bbox_to_anchor=(0.01, 0.01))
    fig.suptitle(
        f"Cube — rotation + translation  |  endpoint-only reconstruction\n"
        f"DSM: {dsm_mean:.2f} dB   Flow Matching: {flow_mean:.2f} dB",
        fontsize=10, y=0.995,
    )

    plt.savefig(OUT_FILE, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
