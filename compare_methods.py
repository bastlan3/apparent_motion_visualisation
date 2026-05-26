"""
Three-way comparison: DSM / Flow Matching / 3D Pose DSM + Langevin equilibrium.

Layout
------
Rows 1-2 : GT and missing input (shared)
Row  3   : DSM reconstruction
Row  4   : Flow matching reconstruction
Row  5   : 3D Pose DSM + Langevin equilibrium (rendered from 3D poses)
Row  6   : 3D time-vector panel (position trajectory + velocity arrows in 3D)

Saves: compare_methods.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from motion_render import ApparentMotionDataset
from motion_denoise import CondUNet, reconstruct_trajectory
from motion_denoise.training import compute_psnr as _psnr
from motion_flow import FlowNet, reconstruct_flow_trajectory
from motion_pose import (
    PoseNet,
    reconstruct_pose_trajectory,
    pose_to_2d_sequence,
)

DSM_CKPT  = "denoiser_best.pt"
FLOW_CKPT = "flow_best.pt"
POSE_CKPT = "pose_best.pt"
OUT_FILE  = "compare_methods.png"
T         = 16
VIS_SEED  = 7


# ── model loaders ─────────────────────────────────────────────────────────────

def load_dsm(device):
    ckpt = torch.load(DSM_CKPT, map_location=device)
    m = CondUNet(base_ch=32, emb_dim=128).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    print(f"DSM  model: {ckpt.get('label','?')}  traj_PSNR={ckpt.get('traj_psnr','?'):.2f} dB")
    return m

def load_flow(device):
    ckpt = torch.load(FLOW_CKPT, map_location=device)
    m = FlowNet(base_ch=32, emb_dim=128).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    print(f"Flow model: traj_PSNR={ckpt.get('traj_psnr','?'):.2f} dB")
    return m

def load_pose(device):
    ckpt = torch.load(POSE_CKPT, map_location=device)
    m = PoseNet(hidden=256, emb_dim=64).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    print(f"Pose model: traj_PSNR={ckpt.get('traj_psnr','?'):.2f} dB")
    return m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dsm_model  = load_dsm(device)
    flow_model = load_flow(device)
    pose_model = load_pose(device)

    ds = ApparentMotionDataset(
        n_sequences=1, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="all_missing", seed=VIS_SEED,
        trans_speed_range=(2.5, 4.0),
        return_poses=True,
    )
    sample     = ds[0]
    targets    = sample["targets"]     # [T, 1, H, W]
    frames     = sample["frames"]      # [T, 1, H, W]
    gt_poses   = sample["poses"]       # [T, 12]
    camera_eye = sample["camera_eye"]  # [3]

    dsm_recon  = reconstruct_trajectory(dsm_model,  targets[0], targets[-1], T, device=device)
    flow_recon = reconstruct_flow_trajectory(flow_model, targets[0], targets[-1], T, device=device)
    pose_recon_poses = reconstruct_pose_trajectory(pose_model, gt_poses[0], gt_poses[-1], T, device=device)
    pose_recon = pose_to_2d_sequence(pose_recon_poses, camera_eye, targets)

    def mid_psnr(recon):
        return [_psnr(recon[t], targets[t]) for t in range(1, T - 1)]

    dsm_ps   = mid_psnr(dsm_recon)
    flow_ps  = mid_psnr(flow_recon)
    pose_ps  = mid_psnr(pose_recon)

    dsm_mean  = sum(dsm_ps)  / len(dsm_ps)
    flow_mean = sum(flow_ps) / len(flow_ps)
    pose_mean = sum(pose_ps) / len(pose_ps)
    print(f"DSM  PSNR: {dsm_mean:.2f} dB")
    print(f"Flow PSNR: {flow_mean:.2f} dB")
    print(f"Pose PSNR: {pose_mean:.2f} dB")

    # ── figure layout ─────────────────────────────────────────────────────────
    # 5 image rows + 3 PSNR strips + 1 tall 3D-panel row
    n_img_rows = 5
    img_cols   = T
    fig_w = T * 1.3
    fig_h = 12.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Upper part: image grid (5 rows × T cols, each with optional PSNR strip)
    gs_top = GridSpec(
        8, T, figure=fig,
        left=0.04, right=0.98, top=0.97, bottom=0.32,
        hspace=0.06, wspace=0.04,
        height_ratios=[1, 1, 1, 0.18, 1, 0.18, 1, 0.18],
    )

    # Lower part: 3D trajectory panel
    gs_bot = GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.98, top=0.28, bottom=0.04,
        wspace=0.3,
    )

    row_cfg = [
        # (gs_row, label,                              seq,        psnrs,     psnr_row, color)
        (0, "Ground truth",                            targets,    None,      None,     None),
        (1, "Input (central missing)",                 frames,     None,      None,     None),
        (2, f"DSM  ({dsm_mean:.1f} dB)",              dsm_recon,  dsm_ps,    3,        "#226633"),
        (4, f"Flow ({flow_mean:.1f} dB)",             flow_recon, flow_ps,   5,        "#22446b"),
        (6, f"3D Pose+Langevin ({pose_mean:.1f} dB)", pose_recon, pose_ps,   7,        "#662244"),
    ]

    for gs_row, label, seq, psnrs, psnr_gs_row, color in row_cfg:
        for t in range(T):
            ax  = fig.add_subplot(gs_top[gs_row, t])
            img = seq[t, 0].numpy().clip(0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])

            is_miss = (gs_row == 1) and (0 < t < T - 1)
            ec = "#cc3333" if is_miss else "#aaaaaa"
            lw = 1.8 if is_miss else 0.5
            for sp in ax.spines.values():
                sp.set_edgecolor(ec); sp.set_linewidth(lw)

            if gs_row == 6:
                ax.set_xlabel(f"t={t}", fontsize=5.5, labelpad=2)

        ax0 = fig.add_subplot(gs_top[gs_row, 0])
        ax0.set_ylabel(label, fontsize=7, labelpad=4)

        if psnrs is not None and psnr_gs_row is not None:
            for t in range(T):
                axp = fig.add_subplot(gs_top[psnr_gs_row, t])
                axp.axis("off")
                if 0 < t < T - 1:
                    axp.text(0.5, 0.6, f"{psnrs[t-1]:.1f}",
                             ha="center", va="center", fontsize=5.5,
                             color=color, transform=axp.transAxes)

    # ── 3D time-vector panel ───────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs_bot[0, 0], projection="3d")

    gt_pos   = gt_poses[:, :3].numpy()
    rec_pos  = pose_recon_poses[:, :3].numpy()

    # GT trajectory
    ax3d.plot(*gt_pos.T,  "o-", color="#555555", lw=1.0, ms=3, label="GT trajectory")
    # Reconstructed trajectory
    ax3d.plot(*rec_pos.T, "s-", color="#cc3333", lw=1.0, ms=3, label="Reconstructed")

    # Time vectors: velocity arrows at each intermediate frame (FD estimate)
    dt_inv = T - 1
    for k in range(1, T - 1):
        p  = rec_pos[k]
        dp = (rec_pos[min(k+1, T-1)] - rec_pos[max(k-1, 0)]) / 2.0 * (1.0 / dt_inv)
        scale = 0.35
        ax3d.quiver(*p, *(dp * scale), color="#22446b", linewidth=0.7, arrow_length_ratio=0.35)

    ax3d.scatter(*gt_pos[0],  c="green",  s=40, zorder=5, label="first frame")
    ax3d.scatter(*gt_pos[-1], c="orange", s=40, zorder=5, label="last frame")

    ax3d.set_xlabel("X", fontsize=7); ax3d.set_ylabel("Y", fontsize=7); ax3d.set_zlabel("Z", fontsize=7)
    ax3d.set_title("3D position trajectory + velocity vectors", fontsize=8)
    ax3d.legend(fontsize=6, loc="upper left")
    ax3d.tick_params(labelsize=6)

    # Angular-velocity magnitude plot (right panel)
    ax_av = fig.add_subplot(gs_bot[0, 1])
    rot_vels = []
    for k in range(1, T - 1):
        R_prev = pose_recon_poses[max(k-1, 0), 3:].reshape(3, 3)
        R_next = pose_recon_poses[min(k+1, T-1), 3:].reshape(3, 3)
        dR = R_next @ R_prev.T
        # Frobenius distance ≈ angular velocity magnitude
        ang_speed = (torch.linalg.matrix_norm(dR - torch.eye(3)) / 2.0).item()
        rot_vels.append(ang_speed)

    ts = list(range(1, T - 1))
    ax_av.bar(ts, rot_vels, color="#662244", alpha=0.7)
    ax_av.set_xlabel("Frame", fontsize=8)
    ax_av.set_ylabel("Angular speed (approx)", fontsize=8)
    ax_av.set_title("Reconstructed angular velocity in 3D", fontsize=8)
    ax_av.tick_params(labelsize=7)

    # ── legend + title ────────────────────────────────────────────────────────
    red_patch = mpatches.Patch(color="#cc3333", label="missing input frame")
    fig.legend(handles=[red_patch], loc="lower left", fontsize=7,
               framealpha=0.85, bbox_to_anchor=(0.01, 0.005))
    fig.suptitle(
        f"Endpoint-only reconstruction  —  cube rotation+translation\n"
        f"DSM: {dsm_mean:.2f} dB   |   Flow: {flow_mean:.2f} dB   |   3D Pose + Langevin: {pose_mean:.2f} dB",
        fontsize=9, y=0.999,
    )

    plt.savefig(OUT_FILE, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
