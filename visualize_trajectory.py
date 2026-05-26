"""
Visualise apparent motion trajectories for four motion modes:
  translate, rotate, both, orbit (off-centre rotation)

For each mode, three rows are shown:
  1. Ground truth (all T frames)
  2. Input        (first + last clean; intermediate frames zeroed)
  3. Reconstruction (model output from endpoints only)

A 3D path panel (right column) shows the body's centroid trajectory in
world space, making the orbit mode's circular arc visually distinct from
the straight-line translate/both paths and the fixed-point rotate case.

Saves trajectory_visualization.png.
"""

import random
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from motion_render import ApparentMotionDataset
from motion_render.scene import Scene
from motion_render.camera import Camera
from motion_render.renderer import Renderer

try:
    from motion_denoise import CondUNet, reconstruct_trajectory
    from motion_denoise.training import compute_psnr
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False

BEST_CKPT = "denoiser_best.pt"
OUT_FILE  = "trajectory_visualization.png"
T         = 16
SEED      = 7


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(device):
    if not HAS_MODEL:
        return None
    try:
        ckpt  = torch.load(BEST_CKPT, map_location=device)
        model = CondUNet(base_ch=32, emb_dim=128).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Loaded model  traj_PSNR={ckpt.get('traj_psnr', '?'):.2f} dB")
        return model
    except Exception as e:
        print(f"Model not loaded ({e}) — reconstruction row will be blank.")
        return None


def collect_3d_path(mode: str, seed_offset: int = 0):
    """Return (positions [T,3], pivot [3] or None) for a given mode."""
    random.seed(SEED + seed_offset)
    torch.manual_seed(SEED + seed_offset)
    scene = Scene.random(n_bodies=1, motion_modes=(mode,))
    body  = scene.bodies[0]
    pivot = body.pivot_point.clone() if body.pivot_point is not None else None
    positions = []
    for _ in range(T):
        positions.append(body.position.clone())
        scene.step(0.05)
    return torch.stack(positions), pivot


def make_sequence(mode: str, seed_offset: int = 0):
    ds = ApparentMotionDataset(
        n_sequences=1, T=T,
        shape_type="cube", motion_mode=mode,
        corruption_mode="all_missing",
        seed=SEED + seed_offset,
        trans_speed_range=(2.5, 4.0),
        rot_speed_range=(1.0, 2.5),
    )
    return ds[0]


def frame_strip(fig, gs_sub, seq, mode_label: str, psnrs=None):
    """Draw T frames in a 1×T grid inside gs_sub."""
    for t in range(T):
        ax  = fig.add_subplot(gs_sub[t])
        img = seq[t, 0].detach().numpy().clip(0, 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel(mode_label, fontsize=7, labelpad=3)

        # PSNR annotation
        if psnrs is not None and 0 < t < T - 1:
            ax.set_title(f"{psnrs[t-1]:.0f}", fontsize=5, pad=1, color="#226633")


def draw_3d_paths(ax3d, path_data):
    """
    path_data: list of (label, positions[T,3], pivot or None, colour)
    """
    for label, pos, pivot, colour in path_data:
        xs, ys, zs = pos[:, 0].numpy(), pos[:, 1].numpy(), pos[:, 2].numpy()
        ax3d.plot(xs, ys, zs, color=colour, linewidth=1.6, label=label)
        # Start marker
        ax3d.scatter(*pos[0].numpy(), color=colour, s=40, marker='o', zorder=5)
        # End marker
        ax3d.scatter(*pos[-1].numpy(), color=colour, s=40, marker='s', zorder=5)
        if pivot is not None:
            ax3d.scatter(*pivot.numpy(), color=colour, s=70, marker='+',
                         linewidths=2, zorder=6)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    modes       = ["translate", "rotate", "both", "orbit"]
    mode_labels = ["Translate", "Rotate", "Both", "Orbit\n(off-centre)"]
    colours     = ["#4488cc", "#cc7733", "#44aa66", "#aa44cc"]

    # ── collect data ──────────────────────────────────────────────────────────
    sequences, recons, psnr_lists, paths = [], [], [], []
    for i, mode in enumerate(modes):
        sample = make_sequence(mode, seed_offset=i * 100)
        sequences.append(sample)

        if model is not None:
            recon = reconstruct_trajectory(
                model, sample["targets"][0], sample["targets"][-1], T, device=device
            )
            psnrs = [compute_psnr(recon[t], sample["targets"][t])
                     for t in range(1, T - 1)]
            mean_p = sum(psnrs) / len(psnrs)
            print(f"  {mode:12s}  mean PSNR = {mean_p:.2f} dB")
        else:
            recon  = torch.zeros_like(sample["targets"])
            psnrs  = None
        recons.append(recon)
        psnr_lists.append(psnrs)

        pos, pivot = collect_3d_path(mode, seed_offset=i * 100)
        paths.append((mode_labels[i].replace("\n", " "), pos, pivot, colours[i]))

    # ── figure layout ─────────────────────────────────────────────────────────
    # Rows: 4 modes × 3 image strips (GT / input / recon), plus 1 right-column 3D panel
    n_modes = len(modes)
    fig_h   = n_modes * 3.8
    fig     = plt.figure(figsize=(T * 1.1 + 3.0, fig_h))

    # Outer grid: left image columns + right 3D panel
    outer = GridSpec(1, 2, figure=fig, width_ratios=[T * 1.1, 3.0],
                     wspace=0.06, left=0.04, right=0.98,
                     top=0.96, bottom=0.04)

    left_gs  = GridSpecFromSubplotSpec(n_modes * 3 + n_modes, 1,
                                       subplot_spec=outer[0],
                                       hspace=0.05,
                                       height_ratios=([1, 1, 1, 0.12] * n_modes))

    # 3D axes in right panel
    ax3d = fig.add_subplot(outer[1], projection='3d')

    row_offset = 0
    for mi, mode in enumerate(modes):
        sample = sequences[mi]
        recon  = recons[mi]
        psnrs  = psnr_lists[mi]

        strip_defs = [
            ("GT",    sample["targets"]),
            ("Input", sample["frames"]),
            ("Recon", recon),
        ]

        for si, (strip_label, seq) in enumerate(strip_defs):
            row_gs = GridSpecFromSubplotSpec(
                1, T,
                subplot_spec=left_gs[row_offset],
                wspace=0.03,
            )
            for t in range(T):
                ax  = fig.add_subplot(row_gs[t])
                img = seq[t, 0].detach().numpy().clip(0, 1)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])

                is_missing = (strip_label == "Input") and (0 < t < T - 1)
                spine_col = "#cc3333" if is_missing else "#bbbbbb"
                spine_lw  = 1.6 if is_missing else 0.4
                for sp in ax.spines.values():
                    sp.set_edgecolor(spine_col); sp.set_linewidth(spine_lw)

                # PSNR on recon row
                if strip_label == "Recon" and psnrs and 0 < t < T - 1:
                    ax.set_title(f"{psnrs[t-1]:.0f}", fontsize=4.5, pad=1,
                                 color="#226633")

                if t == 0:
                    lbl = f"{mode_labels[mi]}\n{strip_label}" if si == 1 else strip_label
                    ax.set_ylabel(lbl, fontsize=6.5, labelpad=3,
                                  color=colours[mi] if si == 1 else "black")

            row_offset += 1

        # spacer row
        row_offset += 1

    # ── 3D path panel ─────────────────────────────────────────────────────────
    draw_3d_paths(ax3d, paths)
    ax3d.set_title("3D centroid paths\n● start  ■ end  + pivot", fontsize=8)
    ax3d.set_xlabel("X", fontsize=7); ax3d.set_ylabel("Y", fontsize=7)
    ax3d.set_zlabel("Z", fontsize=7)
    ax3d.tick_params(labelsize=6)
    ax3d.legend(fontsize=6, loc='upper left')

    fig.suptitle(
        "Apparent motion modes  —  GT / input (missing) / reconstruction",
        fontsize=10, y=0.992,
    )

    plt.savefig(OUT_FILE, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
