import math
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_denoise import CondUNet, train_epoch, val_epoch, reconstruct_trajectory
from motion_denoise.training import compute_psnr

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN         = 500
N_VAL           = 50
T               = 16
BATCH_SIZE      = 8
N_EPOCHS        = 5
LR              = 1e-3
GRAD_CLIP       = 1.0
CHECKPOINT_PATH = "denoiser_checkpoint.pt"


class PrerenderedDataset(Dataset):
    def __init__(self, data: list[dict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pre-render to RAM (renderer is pure Python — slow per-call, fast from cache)
    print(f"\nPre-rendering {N_TRAIN} training sequences (mixed corruption)...")
    train_raw = ApparentMotionDataset(
        n_sequences=N_TRAIN, T=T, corruption_mode="mixed", seed=42
    )
    train_data = [train_raw[i] for i in tqdm(range(N_TRAIN), desc="render train")]

    print(f"Pre-rendering {N_VAL} validation sequences (mixed corruption)...")
    val_raw = ApparentMotionDataset(
        n_sequences=N_VAL, T=T, corruption_mode="mixed", seed=1000
    )
    val_data = [val_raw[i] for i in tqdm(range(N_VAL), desc="render val")]

    # Also pre-render all-missing sequences for the trajectory reconstruction demo
    print(f"Pre-rendering {N_VAL} all-missing sequences for trajectory demo...")
    traj_raw = ApparentMotionDataset(
        n_sequences=N_VAL, T=T, corruption_mode="all_missing", seed=1000
    )
    traj_data = [traj_raw[i] for i in tqdm(range(N_VAL), desc="render traj")]

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

    # Model
    model = CondUNet(base_ch=32, emb_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining for {N_EPOCHS} epochs, batch_size={BATCH_SIZE}\n")
    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=GRAD_CLIP)
        val_loss, val_psnr = val_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{N_EPOCHS} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_PSNR={val_psnr:.2f} dB"
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

    # ── Trajectory reconstruction demo (all intermediate frames missing) ──────
    print("\n" + "=" * 60)
    print("Trajectory reconstruction: all intermediate frames missing")
    print("=" * 60)

    model.eval()
    psnr_list = []

    for sample in traj_data[:10]:   # evaluate on first 10 all-missing sequences
        targets = sample["targets"]   # [T, 1, H, W]  clean ground truth
        ctx_first = targets[0]        # [1, H, W]
        ctx_last  = targets[-1]       # [1, H, W]

        recon = reconstruct_trajectory(model, ctx_first, ctx_last, T, device=device)
        # [T, 1, H, W]

        # PSNR over intermediate frames only
        pred_mid   = recon[1:-1]           # [T-2, 1, H, W]
        target_mid = targets[1:-1]         # [T-2, 1, H, W]
        psnr = compute_psnr(pred_mid, target_mid)
        psnr_list.append(psnr)

    mean_psnr = sum(psnr_list) / len(psnr_list)
    print(f"\nPer-sequence PSNR (intermediate frames vs ground truth):")
    for i, p in enumerate(psnr_list):
        print(f"  sequence {i:02d}: {p:.2f} dB")
    print(f"\nMean PSNR over {len(psnr_list)} sequences: {mean_psnr:.2f} dB")
    print(
        "\nNote: all 14 intermediate frames were reconstructed from only the "
        "first and last clean frames, using temporal position conditioning."
    )


if __name__ == "__main__":
    main()
