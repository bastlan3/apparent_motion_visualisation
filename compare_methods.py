"""
Four-way comparison: DSM / Flow Matching / 3D Pose DSM / ST-DSM (spatiotemporal).

Layout
------
Rows 1-2 : GT and missing input (shared)
Row  3   : DSM reconstruction
Row  4   : Flow matching reconstruction
Row  5   : 3D Pose DSM + Langevin equilibrium (rendered from 3D poses)
Row  6   : ST-DSM + Langevin equilibrium (3D UNet on spatiotemporal volume)
Row  7   : Spacetime diagram (x-t slice at y=H/2) + spatiotemporal vector field

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
from motion_st import STUNet, reconstruct_st_trajectory, st_velocity_field

DSM_CKPT  = "denoiser_best.pt"
FLOW_CKPT = "flow_best.pt"
POSE_CKPT = "pose_best.pt"
ST_CKPT   = "st_best.pt"
OUT_FILE  = "compare_methods.png"
T         = 16
VIS_SEED  = 7


# ── model loaders ─────────────────────────────────────────────────────────────

def load_dsm(device):
    ckpt = torch.load(DSM_CKPT, map_location=device)
    m = CondUNet(base_ch=32, emb_dim=128).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    print(f"DSM  model: traj_PSNR={ckpt.get('traj_psnr','?'):.2f} dB")
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

def load_st(device):
    ckpt = torch.load(ST_CKPT, map_location=device)
    base_ch = ckpt.get("base_ch", 16)
    emb_dim = ckpt.get("emb_dim", 128)
    m = STUNet(base_ch=base_ch, emb_dim=emb_dim).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    print(f"ST   model: traj_PSNR={ckpt.get('traj_psnr','?'):.2f} dB")
    return m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for name, loader in [
        ("dsm",  lambda: load_dsm(device)),
        ("flow", lambda: load_flow(device)),
        ("pose", lambda: load_pose(device)),
        ("st",   lambda: load_st(device)),
    ]:
        try:
            models[name] = loader()
        except FileNotFoundError:
            print(f"  [{name}] checkpoint not found — skipping")

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

    recons = {}
    if "dsm" in models:
        recons["dsm"] = reconstruct_trajectory(models["dsm"], targets[0], targets[-1], T, device=device)
    if "flow" in models:
        recons["flow"] = reconstruct_flow_trajectory(models["flow"], targets[0], targets[-1], T, device=device)
    if "pose" in models:
        pose_recon_poses = reconstruct_pose_trajectory(models["pose"], gt_poses[0], gt_poses[-1], T, device=device)
        recons["pose"] = pose_to_2d_sequence(pose_recon_poses, camera_eye, targets)
    if "st" in models:
        recons["st"] = reconstruct_st_trajectory(models["st"], targets[0:1], targets[-1:], T, device=device)

    def mid_psnr(recon):
        return [_psnr(recon[t], targets[t]) for t in range(1, T - 1)]

    psnrs = {k: mid_psnr(v) for k, v in recons.items()}
    means = {k: sum(v) / len(v) for k, v in psnrs.items()}
    for k, m in means.items():
        print(f"{k.upper():>5} PSNR: {m:.2f} dB")

    # ── figure layout ──────────────────────────────────────────────────────────
    # image rows: GT, Input, DSM, Flow, Pose, ST (with PSNR strips for DSM/Flow/Pose/ST)
    # bottom row: spacetime diagram + spatiotemporal vector field

    n_methods = sum(k in recons for k in ["dsm", "flow", "pose", "st"])
    # height_ratios: GT(1), Input(1), then per method: img(1)+psnr(0.18), final: spacetime(2.5)
    hr = [1, 1]
    for _ in range(n_methods):
        hr += [1, 0.18]
    n_img_rows_gs = 2 + n_methods * 2

    fig_w = T * 1.3
    fig_h = 2 + n_methods * 1.4 + 3.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    top_bottom = 0.28
    gs_top = GridSpec(
        n_img_rows_gs, T, figure=fig,
        left=0.04, right=0.98, top=0.97, bottom=top_bottom,
        hspace=0.06, wspace=0.04,
        height_ratios=hr,
    )
    gs_bot = GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.98, top=top_bottom - 0.02, bottom=0.04,
        wspace=0.3,
    )

    # Build row config dynamically
    row_cfg = [
        (0, "Ground truth",     targets, None, None, None),
        (1, "Input (all miss)", frames,  None, None, None),
    ]
    method_rows = [
        ("dsm",  f"DSM  ({means.get('dsm',0):.1f} dB)",              "#226633"),
        ("flow", f"Flow ({means.get('flow',0):.1f} dB)",              "#22446b"),
        ("pose", f"3D Pose+Langevin ({means.get('pose',0):.1f} dB)",  "#662244"),
        ("st",   f"ST-DSM+Langevin ({means.get('st',0):.1f} dB)",    "#336655"),
    ]
    gs_row = 2
    for key, label, color in method_rows:
        if key not in recons:
            continue
        row_cfg.append((gs_row, label, recons[key], psnrs[key], gs_row + 1, color))
        gs_row += 2

    for gs_r, label, seq, psnr_list, psnr_gs_row, color in row_cfg:
        for t in range(T):
            ax  = fig.add_subplot(gs_top[gs_r, t])
            img = seq[t, 0].numpy().clip(0, 1)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])

            is_miss = (gs_r == 1) and (0 < t < T - 1)
            ec = "#cc3333" if is_miss else "#aaaaaa"
            lw = 1.8 if is_miss else 0.5
            for sp in ax.spines.values():
                sp.set_edgecolor(ec); sp.set_linewidth(lw)

            if gs_r == gs_row - 2:   # last image row
                ax.set_xlabel(f"t={t}", fontsize=5.5, labelpad=2)

        ax0 = fig.add_subplot(gs_top[gs_r, 0])
        ax0.set_ylabel(label, fontsize=7, labelpad=4)

        if psnr_list is not None and psnr_gs_row is not None:
            for t in range(T):
                axp = fig.add_subplot(gs_top[psnr_gs_row, t])
                axp.axis("off")
                if 0 < t < T - 1:
                    axp.text(0.5, 0.6, f"{psnr_list[t-1]:.1f}",
                             ha="center", va="center", fontsize=5.5,
                             color=color, transform=axp.transAxes)

    # ── Bottom left: spacetime diagram (x-t slice at y = H/2) ─────────────────
    ax_st = fig.add_subplot(gs_bot[0, 0])

    if "st" in recons:
        seq_st = recons["st"]   # [T, 1, H, W]
        H_img  = seq_st.shape[-2]
        y_mid  = H_img // 2

        # x-t image: each column is a frame, rows are x pixels
        xt_img = seq_st[:, 0, y_mid, :].numpy().T  # [W, T]
        ax_st.imshow(xt_img, cmap="gray", vmin=0, vmax=1,
                     aspect="auto", origin="lower",
                     extent=[0, T-1, 0, seq_st.shape[-1]-1])

        # Overlay spatiotemporal gradient arrows (sub-sampled)
        I_x, I_y, I_t = st_velocity_field(seq_st)
        # Take the y=H/2-1 row (accounting for interior trimming)
        y_row = y_mid - 1
        y_row = max(0, min(y_row, I_x.shape[1] - 1))
        step  = 4
        for ti in range(0, I_t.shape[0], step):
            for xi in range(0, I_t.shape[2], step):
                vt = I_t[ti, y_row, xi].item()
                vx = I_x[ti, y_row, xi].item()
                mag = math.sqrt(vt**2 + vx**2)
                if mag > 0.02:
                    scale = 2.0 / (mag + 1e-6)
                    ax_st.annotate(
                        "", xy=(ti + 1 + vt * scale, xi + vx * scale),
                        xytext=(ti + 1, xi),
                        arrowprops=dict(arrowstyle="->", color="#cc3333",
                                        lw=0.6, mutation_scale=5),
                    )

        ax_st.set_xlabel("Time (frame index)", fontsize=8)
        ax_st.set_ylabel("x pixel", fontsize=8)
        ax_st.set_title("Spacetime slice (y=H/2) + gradient vectors", fontsize=8)
        ax_st.tick_params(labelsize=6)
    else:
        ax_st.text(0.5, 0.5, "ST model not loaded", ha="center", va="center",
                   transform=ax_st.transAxes, fontsize=9)
        ax_st.axis("off")

    # ── Bottom right: PSNR-vs-time comparison across methods ──────────────────
    ax_psnr = fig.add_subplot(gs_bot[0, 1])
    colors_map = {
        "dsm":  "#226633",
        "flow": "#22446b",
        "pose": "#662244",
        "st":   "#336655",
    }
    labels_map = {
        "dsm":  "DSM",
        "flow": "Flow",
        "pose": "3D Pose",
        "st":   "ST-DSM",
    }
    ts = list(range(1, T - 1))
    for key, color in colors_map.items():
        if key in psnrs:
            ax_psnr.plot(ts, psnrs[key], "o-", color=color,
                         lw=1.2, ms=3, label=f"{labels_map[key]} ({means[key]:.1f} dB)")
    ax_psnr.set_xlabel("Frame index", fontsize=8)
    ax_psnr.set_ylabel("PSNR (dB)", fontsize=8)
    ax_psnr.set_title("Per-frame PSNR (all methods)", fontsize=8)
    ax_psnr.legend(fontsize=6, loc="lower center")
    ax_psnr.tick_params(labelsize=6)

    # ── title & legend ────────────────────────────────────────────────────────
    red_patch = mpatches.Patch(color="#cc3333", label="missing input frame")
    fig.legend(handles=[red_patch], loc="lower left", fontsize=7,
               framealpha=0.85, bbox_to_anchor=(0.01, 0.005))

    title_parts = [f"{labels_map[k]}: {means[k]:.2f} dB" for k in ["dsm","flow","pose","st"] if k in means]
    fig.suptitle(
        "Endpoint-only reconstruction — cube rotation+translation\n" + "   |   ".join(title_parts),
        fontsize=9, y=0.999,
    )

    plt.savefig(OUT_FILE, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FILE}")


import math

if __name__ == "__main__":
    main()
