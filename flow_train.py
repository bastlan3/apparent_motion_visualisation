"""
Train a FlowNet on cube rotation+translation sequences and compare it to
the best DSM denoiser from experiment_grid.py.

Flow matching trains the model to predict the velocity field
  v_θ(x_t, t; ctx_first, ctx_last) ≈ dx/dt
at every point in (image × time) space.  At inference the ODE is integrated
with the Heun method to reconstruct all intermediate frames from the two
endpoint frames only.

Saves: flow_best.pt
"""

import math
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_flow import FlowNet, train_flow_epoch, val_flow_epoch, reconstruct_flow_trajectory, compute_psnr

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN      = 200
N_VAL        = 40
N_TRAJ_EVAL  = 10
T            = 16
BATCH_SIZE   = 8
N_EPOCHS     = 20
LR           = 1e-3
GRAD_CLIP    = 1.0
BEST_CKPT    = "flow_best.pt"


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
        recon   = reconstruct_flow_trajectory(model, targets[0], targets[-1], T, device=device)
        psnrs.append(compute_psnr(recon[1:-1], targets[1:-1]))
    return sum(psnrs) / len(psnrs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Shape: cube | Motion: both (rotation+translation)")
    print(f"N_TRAIN={N_TRAIN}, N_EPOCHS={N_EPOCHS}\n")

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

    model = FlowNet(base_ch=32, emb_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFlowNet parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state    = None

    print()
    for epoch in range(1, N_EPOCHS + 1):
        tr_loss = train_flow_epoch(model, train_loader, optimizer, device, grad_clip=GRAD_CLIP)
        val_loss, val_psnr = val_flow_epoch(model, val_loader, device)
        scheduler.step()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        print(
            f"  ep {epoch:02d}/{N_EPOCHS} | "
            f"tr={tr_loss:.4f} | val={val_loss:.4f} | step-PSNR={val_psnr:.1f} dB{marker}"
        )

    model.load_state_dict(best_state)
    traj_psnr = eval_trajectory_psnr(model, traj_data, N_TRAJ_EVAL, device)
    print(f"\nTrajectory PSNR (all-missing, Heun ODE): {traj_psnr:.2f} dB")

    torch.save(
        {
            "model_state": best_state,
            "val_loss":    best_val_loss,
            "traj_psnr":   traj_psnr,
        },
        BEST_CKPT,
    )
    print(f"Saved: {BEST_CKPT}")

    # ── Compare against the DSM baseline ─────────────────────────────────────
    dsm_ckpt = "denoiser_best.pt"
    try:
        from motion_denoise import CondUNet, reconstruct_trajectory
        from motion_denoise.training import compute_psnr as dsm_psnr

        ckpt_dsm  = torch.load(dsm_ckpt, map_location=device)
        dsm_model = CondUNet(base_ch=32, emb_dim=128).to(device)
        dsm_model.load_state_dict(ckpt_dsm["model_state"])
        dsm_model.eval()

        dsm_traj_psnrs = []
        for sample in traj_data[:N_TRAJ_EVAL]:
            targets = sample["targets"]
            recon   = reconstruct_trajectory(dsm_model, targets[0], targets[-1], T, device=device)
            dsm_traj_psnrs.append(dsm_psnr(recon[1:-1], targets[1:-1]))
        dsm_psnr_val = sum(dsm_traj_psnrs) / len(dsm_traj_psnrs)

        print(f"\n{'='*50}")
        print(f"{'Method':<30} {'traj PSNR':>10}")
        print(f"{'-'*50}")
        print(f"  {'DSM (cosine+gaussian)':<28} {dsm_psnr_val:>9.2f} dB")
        print(f"  {'Flow Matching (Heun ODE)':<28} {traj_psnr:>9.2f} dB")
        print(f"{'='*50}")
    except FileNotFoundError:
        print(f"\n(Skipping DSM comparison — {dsm_ckpt} not found)")


if __name__ == "__main__":
    main()
