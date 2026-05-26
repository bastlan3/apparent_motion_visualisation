"""
DSM training in 3D pose space with the same curriculum as motion_denoise.

Loss:
  t_idx ~ Uniform{1..T-2}
  σ ~ log-Uniform[σ_min, σ_max]
  pose_noisy = pose_t + σ · ε,   ε ~ N(0, I_{12})
  loss = MSE(PoseNet(pose_noisy, σ, pose_0, pose_T, t/(T-1)), pose_t)

Curriculum:
  sigma_max is annealed from sigma_start to sigma_target (cosine by default)
  missing_frac fraction of each batch use pose_noisy=0 (all-missing regime)
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


def pose_dsm_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
    missing_frac: float = 0.0,
    missing_sigma: float = 0.8,
) -> Tensor:
    poses = batch["poses"].to(device)   # [B, T, 12]
    B, T  = poses.shape[:2]
    assert T >= 3

    pose_0 = poses[:, 0]    # [B, 12]
    pose_T = poses[:, -1]   # [B, 12]

    t_idx   = torch.randint(1, T - 1, (B,), device=device)
    pose_t  = poses[torch.arange(B), t_idx]             # [B, 12]
    t_frac  = t_idx.float() / (T - 1)                   # [B]

    log_sigma = (
        torch.rand(B, device=device) * (math.log(sigma_max) - math.log(sigma_min))
        + math.log(sigma_min)
    )
    sigma = torch.exp(log_sigma)
    pose_noisy = pose_t + sigma[:, None] * torch.randn_like(pose_t)

    if missing_frac > 0.0:
        n_miss   = max(1, int(B * missing_frac))
        miss_idx = torch.randperm(B, device=device)[:n_miss]
        pose_noisy          = pose_noisy.clone()
        sigma               = sigma.clone()
        pose_noisy[miss_idx] = 0.0
        sigma[miss_idx]      = missing_sigma

    pose_pred = model(pose_noisy, sigma, pose_0, pose_T, t_frac)
    return F.mse_loss(pose_pred, pose_t)


def compute_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.detach(), target.detach()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def train_pose_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    sigma_max: float = 1.5,
    missing_frac: float = 0.0,
    missing_sigma: float = 0.8,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = pose_dsm_loss(model, batch, device,
                             sigma_max=sigma_max,
                             missing_frac=missing_frac,
                             missing_sigma=missing_sigma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_pose_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Validates on clean GT poses (sigma ~ log-Uniform, no missing frac)."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            poses  = batch["poses"].to(device)
            B, T   = poses.shape[:2]
            pose_0 = poses[:, 0]
            pose_T = poses[:, -1]

            t_idx   = torch.randint(1, T - 1, (B,), device=device)
            pose_t  = poses[torch.arange(B), t_idx]
            t_frac  = t_idx.float() / (T - 1)

            log_sigma = (
                torch.rand(B, device=device) * (math.log(1.5) - math.log(0.01))
                + math.log(0.01)
            )
            sigma      = torch.exp(log_sigma)
            pose_noisy = pose_t + sigma[:, None] * torch.randn_like(pose_t)

            pose_pred = model(pose_noisy, sigma, pose_0, pose_T, t_frac)
            loss = F.mse_loss(pose_pred, pose_t)
            total_loss += loss.item()

            # PSNR proxy: position error in normalised units
            from .pose_net import POSE_SCALE
            pos_pred   = pose_pred[:, :3] / POSE_SCALE
            pos_target = pose_t[:, :3]    / POSE_SCALE
            mse_pos    = F.mse_loss(pos_pred, pos_target).item()
            if mse_pos > 0:
                total_psnr += 10.0 * math.log10(1.0 / mse_pos)

    n = len(loader)
    return total_loss / n, total_psnr / n
