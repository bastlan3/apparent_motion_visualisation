import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from motion_render import ApparentMotionDataset, sequence_collate_fn
from motion_denoise import CondUNet, train_epoch, val_epoch

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
    print(f"\nPre-rendering {N_TRAIN} training sequences...")
    train_raw = ApparentMotionDataset(
        n_sequences=N_TRAIN, T=T, corruption_mode="mixed", seed=42
    )
    train_data = [train_raw[i] for i in tqdm(range(N_TRAIN), desc="render train")]

    print(f"Pre-rendering {N_VAL} validation sequences...")
    val_raw = ApparentMotionDataset(
        n_sequences=N_VAL, T=T, corruption_mode="mixed", seed=1000
    )
    val_data = [val_raw[i] for i in tqdm(range(N_VAL), desc="render val")]

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


if __name__ == "__main__":
    main()
