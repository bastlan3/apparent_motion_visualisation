"""
Train a spatiotemporal 3D UNet (STUNet) with DSM on (H × W × T) volumes.

The full sequence of T-2 intermediate frames is treated as a single 3D volume
so the model learns joint spatiotemporal correlations rather than per-frame scores.
At inference, annealed Langevin dynamics equilibrate the entire intermediate volume
simultaneously.

Saves: st_best.pt
"""

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_st import (
    STUNet,
    train_st_epoch,
    val_st_epoch,
    compute_psnr,
    sigma_max_schedule,
    reconstruct_st_trajectory,
)
from motion_denoise.training import compute_psnr as img_psnr

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN       = 200
N_VAL         = 40
N_TRAJ_EVAL   = 10
T             = 16
BATCH_SIZE    = 4
N_EPOCHS      = 20
LR            = 1e-3
GRAD_CLIP     = 1.0
SIGMA_START   = 0.15
SIGMA_TARGET  = 1.5
WARMUP_FRAC   = 0.67
MISSING_FRAC  = 0.25
BASE_CH       = 16
EMB_DIM       = 128
BEST_CKPT     = "st_best.pt"


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
        targets    = sample["targets"]   # [T, 1, H, W]
        ctx_first  = targets[0:1]
        ctx_last   = targets[-1:]
        recon = reconstruct_st_trajectory(model, ctx_first, ctx_last, T, device=device)
        mid_psnr = img_psnr(recon[1:-1], targets[1:-1])
        psnrs.append(mid_psnr)
    return sum(psnrs) / len(psnrs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Shape: cube | Motion: both  |  N_TRAIN={N_TRAIN}, N_EPOCHS={N_EPOCHS}")
    print(f"Curriculum: σ_max cosine {SIGMA_START}→{SIGMA_TARGET}, missing_frac={MISSING_FRAC}")
    print(f"STUNet base_ch={BASE_CH}, emb_dim={EMB_DIM}\n")

    print("Pre-rendering training data...")
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

    print("Pre-rendering trajectory eval data...")
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

    model = STUNet(base_ch=BASE_CH, emb_dim=EMB_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSTUNet parameters: {n_params:,}\n")

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
        tr_loss = train_st_epoch(
            model, train_loader, optimizer, device,
            grad_clip=GRAD_CLIP, sigma_max=smax,
            missing_frac=MISSING_FRAC,
        )
        val_loss, val_psnr = val_st_epoch(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        print(
            f"  ep {epoch:02d}/{N_EPOCHS} | σ_max={smax:.3f} | "
            f"tr={tr_loss:.4f} | val={val_loss:.4f} | vol-PSNR={val_psnr:.1f} dB{marker}"
        )

    model.load_state_dict(best_state)
    traj_psnr = eval_trajectory_psnr(model, traj_data, N_TRAJ_EVAL, device)
    print(f"\nTrajectory PSNR (Langevin spatiotemporal equilibrium): {traj_psnr:.2f} dB")

    torch.save({
        "model_state": best_state,
        "val_loss":    best_val_loss,
        "traj_psnr":   traj_psnr,
        "base_ch":     BASE_CH,
        "emb_dim":     EMB_DIM,
    }, BEST_CKPT)
    print(f"Saved: {BEST_CKPT}")

    # ── comparison table ──────────────────────────────────────────────────────
    rows = [("ST-DSM  3D volume + Langevin equilibrium", traj_psnr)]
    for ckpt_path, label, loader_fn in [
        ("denoiser_best.pt", "DSM  cosine+gaussian (curriculum)",
         lambda: _load_dsm(ckpt_path, device, traj_data, T)),
        ("flow_best.pt",     "Flow cosine curriculum + missing",
         lambda: _load_flow(ckpt_path, device, traj_data, T)),
        ("pose_best.pt",     "3D Pose DSM + Langevin",
         lambda: _load_pose(ckpt_path, device, traj_data, T)),
    ]:
        try:
            rows.append((label, loader_fn()))
        except FileNotFoundError:
            pass

    print(f"\n{'='*62}")
    print(f"{'Method':<44} {'traj PSNR':>10}")
    print(f"{'-'*62}")
    for label, psnr in sorted(rows, key=lambda r: -r[1]):
        print(f"  {label:<42} {psnr:>9.2f} dB")
    print(f"{'='*62}")


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


def _load_pose(path, device, traj_data, T):
    from motion_pose import PoseNet, reconstruct_pose_trajectory, pose_to_2d_sequence
    from motion_denoise.training import compute_psnr as cp
    ckpt = torch.load(path, map_location=device)
    m = PoseNet(hidden=256, emb_dim=64).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    psnrs = []
    for s in traj_data:
        tgt  = s["targets"]
        eye  = s["camera_eye"]
        poses = s["poses"]
        rec  = pose_to_2d_sequence(
            reconstruct_pose_trajectory(m, poses[0], poses[-1], T, device=device),
            eye, tgt,
        )
        psnrs.append(cp(rec[1:-1], tgt[1:-1]))
    return sum(psnrs) / len(psnrs)


if __name__ == "__main__":
    main()
