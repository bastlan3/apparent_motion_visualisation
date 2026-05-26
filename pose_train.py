"""
Train a PoseNet with DSM in 3D pose space using the same curriculum as the
best DSM image-space model.

The model works in the 12D rigid-body pose space (3 position + 9 orientation)
rather than the 4096D image space.  At inference, annealed Langevin dynamics
sample the equilibrium distribution p(pose_t | pose_0, pose_T), then each
pose is rendered to 2D with the known wireframe renderer.

Saves: pose_best.pt
"""

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset
from motion_pose import (
    PoseNet,
    train_pose_epoch,
    val_pose_epoch,
    compute_psnr,
    sigma_max_schedule,
    reconstruct_pose_trajectory,
    pose_to_2d_sequence,
)

# ── Config (mirrors experiment_grid.py) ───────────────────────────────────────
N_TRAIN        = 200
N_VAL          = 40
N_TRAJ_EVAL    = 10
T              = 16
BATCH_SIZE     = 8
N_EPOCHS       = 20
LR             = 1e-3
GRAD_CLIP      = 1.0
SIGMA_START    = 0.15
SIGMA_TARGET   = 1.5
WARMUP_FRAC    = 0.67
MISSING_FRAC   = 0.25
BEST_CKPT      = "pose_best.pt"

# Rendering loss: turned on after RENDER_WARMUP_EPOCH so the DSM loss
# has time to produce reasonable poses before pixel gradients are added.
RENDER_LAMBDA       = 5.0    # pixel loss weight (images are in [0,1]²)
RENDER_WARMUP_EPOCH = 8      # epochs before render loss is added


class PrerenderedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def pose_collate_fn(batch):
    return {
        "poses":      torch.stack([b["poses"]      for b in batch]),
        "targets":    torch.stack([b["targets"]    for b in batch]),
        "frames":     torch.stack([b["frames"]     for b in batch]),
        "mask":       torch.stack([b["mask"]       for b in batch]),
        "camera_eye": torch.stack([b["camera_eye"] for b in batch]),
    }


def eval_trajectory_psnr(model, traj_data, n, device):
    model.eval()
    psnrs = []
    for sample in traj_data[:n]:
        targets    = sample["targets"]   # [T, 1, H, W]
        poses      = sample["poses"]     # [T, 12]
        camera_eye = sample["camera_eye"]

        recon_poses = reconstruct_pose_trajectory(
            model, poses[0], poses[-1], T, device=device
        )
        recon_frames = pose_to_2d_sequence(recon_poses, camera_eye, targets)
        psnrs.append(compute_psnr(recon_frames[1:-1], targets[1:-1]))
    return sum(psnrs) / len(psnrs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Shape: cube | Motion: both  |  N_TRAIN={N_TRAIN}, N_EPOCHS={N_EPOCHS}")
    print(f"Curriculum: σ_max cosine {SIGMA_START}→{SIGMA_TARGET}, missing_frac={MISSING_FRAC}\n")

    print("Pre-rendering training data (with 3D poses)...")
    ds_train = ApparentMotionDataset(
        n_sequences=N_TRAIN, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="mixed", seed=42,
        return_poses=True,
    )
    train_data = [ds_train[i] for i in tqdm(range(N_TRAIN), desc="train")]

    print("Pre-rendering validation data...")
    ds_val = ApparentMotionDataset(
        n_sequences=N_VAL, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="mixed", seed=1000,
        return_poses=True,
    )
    val_data = [ds_val[i] for i in tqdm(range(N_VAL), desc="val")]

    print("Pre-rendering trajectory eval data...")
    ds_traj = ApparentMotionDataset(
        n_sequences=N_TRAJ_EVAL, T=T,
        shape_type="cube", motion_mode="both",
        corruption_mode="all_missing", seed=1000,
        return_poses=True,
    )
    traj_data = [ds_traj[i] for i in tqdm(range(N_TRAJ_EVAL), desc="traj")]

    train_loader = DataLoader(
        PrerenderedDataset(train_data),
        batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=pose_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        PrerenderedDataset(val_data),
        batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=pose_collate_fn, num_workers=0,
    )

    model = PoseNet(hidden=256, emb_dim=64).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPoseNet parameters: {n_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, N_EPOCHS + 1):
        smax = sigma_max_schedule(
            epoch, N_EPOCHS, SIGMA_START, SIGMA_TARGET,
            schedule="cosine", warmup_frac=WARMUP_FRAC,
        )
        rl = RENDER_LAMBDA if epoch > RENDER_WARMUP_EPOCH else 0.0
        tr_loss = train_pose_epoch(
            model, train_loader, optimizer, device,
            grad_clip=GRAD_CLIP, sigma_max=smax,
            missing_frac=MISSING_FRAC, render_lambda=rl,
        )
        val_loss, val_psnr = val_pose_epoch(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        rl_tag = f" +render(λ={rl:.0f})" if rl > 0 else ""
        print(
            f"  ep {epoch:02d}/{N_EPOCHS} | σ_max={smax:.3f}{rl_tag} | "
            f"tr={tr_loss:.4f} | val={val_loss:.4f} | pos-PSNR={val_psnr:.1f} dB{marker}"
        )

    model.load_state_dict(best_state)
    traj_psnr = eval_trajectory_psnr(model, traj_data, N_TRAJ_EVAL, device)
    print(f"\nTrajectory PSNR (render from Langevin 3D poses): {traj_psnr:.2f} dB")

    torch.save({"model_state": best_state, "val_loss": best_val_loss,
                "traj_psnr": traj_psnr}, BEST_CKPT)
    print(f"Saved: {BEST_CKPT}")

    # ── comparison table ─────────────────────────────────────────────────────
    rows = [("3D Pose DSM + Langevin equilibrium", traj_psnr)]
    for ckpt_path, label, loader_fn in [
        ("denoiser_best.pt",  "DSM  cosine+gaussian  (curriculum)",
         lambda: _load_dsm(ckpt_path, device, traj_data, T)),
        ("flow_best.pt",      "Flow cosine curriculum + missing",
         lambda: _load_flow(ckpt_path, device, traj_data, T)),
    ]:
        try:
            rows.append((label, loader_fn()))
        except FileNotFoundError:
            pass

    print(f"\n{'='*58}")
    print(f"{'Method':<40} {'traj PSNR':>10}")
    print(f"{'-'*58}")
    for label, psnr in sorted(rows, key=lambda r: -r[1]):
        print(f"  {label:<38} {psnr:>9.2f} dB")
    print(f"{'='*58}")


def _load_dsm(path, device, traj_data, T):
    from motion_denoise import CondUNet, reconstruct_trajectory
    from motion_denoise.training import compute_psnr as cp
    ckpt = torch.load(path, map_location=device)
    m = CondUNet(base_ch=32, emb_dim=128).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    psnrs = []
    for s in traj_data:
        tgt = s["targets"]
        psnrs.append(cp(reconstruct_trajectory(m, tgt[0], tgt[-1], T, device=device)[1:-1], tgt[1:-1]))
    return sum(psnrs) / len(psnrs)


def _load_flow(path, device, traj_data, T):
    from motion_flow import FlowNet, reconstruct_flow_trajectory, compute_psnr as cp
    ckpt = torch.load(path, map_location=device)
    m = FlowNet(base_ch=32, emb_dim=128).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    psnrs = []
    for s in traj_data:
        tgt = s["targets"]
        psnrs.append(cp(reconstruct_flow_trajectory(m, tgt[0], tgt[-1], T, device=device)[1:-1], tgt[1:-1]))
    return sum(psnrs) / len(psnrs)


if __name__ == "__main__":
    main()
