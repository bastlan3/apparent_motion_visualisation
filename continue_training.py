import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_denoise import CondUNet, train_epoch, val_epoch, reconstruct_trajectory
from motion_denoise.training import compute_psnr

N_TRAIN         = 500
N_VAL           = 50
T               = 16
BATCH_SIZE      = 8
N_MORE_EPOCHS   = 25
LR_RESUME       = 3e-4   # lower than initial 1e-3; previous schedule ended near 1e-5
GRAD_CLIP       = 1.0
CHECKPOINT_PATH = "denoiser_checkpoint.pt"


class PrerenderedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    start_epoch = ckpt["epoch"]
    print(f"Resuming from epoch {start_epoch} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_PSNR={ckpt['val_psnr']:.2f} dB)")

    model = CondUNet(base_ch=32, emb_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Pre-render (same seeds → same data as original run)
    print(f"\nPre-rendering {N_TRAIN} training sequences...")
    train_data = [
        ApparentMotionDataset(n_sequences=N_TRAIN, T=T, corruption_mode="mixed", seed=42)[i]
        for i in tqdm(range(N_TRAIN), desc="render train")
    ]
    print(f"Pre-rendering {N_VAL} validation sequences...")
    val_data = [
        ApparentMotionDataset(n_sequences=N_VAL, T=T, corruption_mode="mixed", seed=1000)[i]
        for i in tqdm(range(N_VAL), desc="render val")
    ]
    print(f"Pre-rendering {N_VAL} all-missing sequences for trajectory demo...")
    traj_data = [
        ApparentMotionDataset(n_sequences=N_VAL, T=T, corruption_mode="all_missing", seed=1000)[i]
        for i in tqdm(range(N_VAL), desc="render traj")
    ]

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

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RESUME)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_MORE_EPOCHS, eta_min=1e-6
    )

    print(f"\nContinuing for {N_MORE_EPOCHS} more epochs "
          f"(epochs {start_epoch+1}–{start_epoch+N_MORE_EPOCHS})\n")

    best_val_loss = ckpt["val_loss"]

    for i in range(1, N_MORE_EPOCHS + 1):
        epoch = start_epoch + i
        tr_loss = train_epoch(model, train_loader, optimizer, device, grad_clip=GRAD_CLIP)
        val_loss, val_psnr = val_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
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

    print(f"\nDone. Best val_loss={best_val_loss:.4f}")

    # ── Trajectory reconstruction demo ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Trajectory reconstruction: all intermediate frames missing")
    print("=" * 60)

    model.eval()
    psnr_list = []
    for sample in traj_data[:10]:
        targets   = sample["targets"]          # [T, 1, H, W]
        recon     = reconstruct_trajectory(model, targets[0], targets[-1], T, device=device)
        psnr      = compute_psnr(recon[1:-1], targets[1:-1])
        psnr_list.append(psnr)

    mean_psnr = sum(psnr_list) / len(psnr_list)
    print(f"\nPer-sequence PSNR (intermediate frames vs ground truth):")
    for i, p in enumerate(psnr_list):
        print(f"  sequence {i:02d}: {p:.2f} dB")
    print(f"\nMean PSNR over {len(psnr_list)} sequences: {mean_psnr:.2f} dB")


if __name__ == "__main__":
    main()
