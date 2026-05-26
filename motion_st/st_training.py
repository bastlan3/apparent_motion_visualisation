"""
DSM training on spatiotemporal volumes with the same curriculum as motion_denoise.

Total loss = MSE(STUNet(vol_noisy, σ, ctx_first, ctx_last), vol_clean)

The full sequence of T-2 intermediate frames is corrupted at once with a
single σ per sample, making the score network learn joint spatiotemporal
denoising rather than per-frame denoising.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


def sigma_max_schedule(
    epoch: int,
    total_epochs: int,
    sigma_start: float = 0.15,
    sigma_target: float = 1.5,
    schedule: str = "cosine",
    warmup_frac: float = 0.67,
) -> float:
    warmup = max(1, int(total_epochs * warmup_frac))
    t = min(epoch / warmup, 1.0)
    if schedule == "cosine":
        t = (1.0 - math.cos(math.pi * t)) / 2.0
    elif schedule == "exp":
        t = (math.exp(t) - 1.0) / (math.e - 1.0)
    return sigma_start + (sigma_target - sigma_start) * t


def st_dsm_loss(
    model,
    batch:         dict,
    device:        torch.device,
    sigma_min:     float = 0.01,
    sigma_max:     float = 1.5,
    missing_frac:  float = 0.0,
    missing_sigma: float = 0.8,
) -> Tensor:
    targets = batch["targets"].to(device)   # [B, T, 1, H, W]
    B, T, C, H, W = targets.shape
    assert T >= 3, "Need at least 3 frames"

    # ctx endpoints are always clean
    ctx_first = targets[:, 0:1]    # [B, 1, 1, H, W]
    ctx_last  = targets[:, -1:]    # [B, 1, 1, H, W]
    Tm = T - 2

    # Intermediate clean volume: [B, 1, Tm, H, W]
    vol_clean = targets[:, 1:-1].permute(0, 2, 1, 3, 4)

    # Single σ per sample
    log_sigma = (
        torch.rand(B, device=device) * (math.log(sigma_max) - math.log(sigma_min))
        + math.log(sigma_min)
    )
    sigma     = torch.exp(log_sigma)
    vol_noisy = vol_clean + sigma[:, None, None, None, None] * torch.randn_like(vol_clean)

    if missing_frac > 0.0:
        n_miss   = max(1, int(B * missing_frac))
        miss_idx = torch.randperm(B, device=device)[:n_miss]
        vol_noisy         = vol_noisy.clone()
        sigma             = sigma.clone()
        vol_noisy[miss_idx] = 0.0
        sigma[miss_idx]     = missing_sigma

    # Broadcast ctx across Tm
    ctx_f = ctx_first.permute(0, 2, 1, 3, 4).expand(-1, -1, Tm, -1, -1)  # [B, 1, Tm, H, W]
    ctx_l = ctx_last.permute(0, 2, 1, 3, 4).expand(-1, -1, Tm, -1, -1)

    vol_pred = model(vol_noisy, sigma, ctx_f, ctx_l)                        # [B, 1, Tm, H, W]
    return F.mse_loss(vol_pred, vol_clean)


def compute_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.detach(), target.detach()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def train_st_epoch(
    model,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    device:        torch.device,
    grad_clip:     float = 1.0,
    sigma_max:     float = 1.5,
    missing_frac:  float = 0.0,
    missing_sigma: float = 0.8,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = st_dsm_loss(
            model, batch, device,
            sigma_max=sigma_max,
            missing_frac=missing_frac,
            missing_sigma=missing_sigma,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_st_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)   # [B, T, 1, H, W]
            B, T, C, H, W = targets.shape
            Tm = T - 2

            ctx_f = targets[:, 0:1].permute(0, 2, 1, 3, 4).expand(-1, -1, Tm, -1, -1)
            ctx_l = targets[:, -1:].permute(0, 2, 1, 3, 4).expand(-1, -1, Tm, -1, -1)
            vol_clean = targets[:, 1:-1].permute(0, 2, 1, 3, 4)

            log_sigma = (
                torch.rand(B, device=device) * (math.log(1.5) - math.log(0.01))
                + math.log(0.01)
            )
            sigma     = torch.exp(log_sigma)
            vol_noisy = vol_clean + sigma[:, None, None, None, None] * torch.randn_like(vol_clean)

            vol_pred = model(vol_noisy, sigma, ctx_f, ctx_l)
            loss     = F.mse_loss(vol_pred, vol_clean)
            total_loss += loss.item()
            total_psnr += compute_psnr(vol_pred, vol_clean)

    n = len(loader)
    return total_loss / n, total_psnr / n
