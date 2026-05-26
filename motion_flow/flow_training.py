"""
Flow matching training utilities.

Loss: conditional flow matching on ground-truth trajectories.

For a sequence x_0, x_1, ..., x_{T-1} the target velocity at frame t is the
central finite-difference approximation of dx/dt:

    v*(t) = (x_{t+1} - x_{t-1}) / (2 * dt_norm)
    dt_norm = 1 / (T - 1)

The model learns  v_θ(x_t, t/(T-1), ctx_first, ctx_last) ≈ v*(t).

Curriculum (mirrors the DSM curriculum in motion_denoise/training.py):
  sigma_max   – log-uniform noise added to x_t, annealing from a small value
                to sigma_max; simulates ODE drift from the GT trajectory.
  missing_frac – fraction of each batch where x_t is replaced with zeros,
                 exactly matching the all-missing inference scenario.
                 The velocity target is unchanged (still the GT FD velocity),
                 so the model learns to predict correct velocities even when it
                 cannot see x_t and must rely on ctx_first, ctx_last, and t.
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


def flow_matching_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 0.0,
    missing_frac: float = 0.0,
) -> Tensor:
    targets = batch["targets"].to(device)   # [B, T, 1, H, W]
    B, T = targets.shape[:2]
    assert T >= 3, f"T must be >= 3, got {T}"

    ctx_first = targets[:, 0]    # [B, 1, H, W]
    ctx_last  = targets[:, -1]   # [B, 1, H, W]

    # Sample a random interior frame (need neighbours for FD)
    t_idx = torch.randint(1, T - 1, (B,), device=device)   # ∈ {1..T-2}

    t_norm = t_idx.float() / (T - 1)                       # ∈ (0, 1)
    x_t    = targets[torch.arange(B), t_idx]
    x_prev = targets[torch.arange(B), t_idx - 1]
    x_next = targets[torch.arange(B), t_idx + 1]

    # Curriculum noise on x_t: simulates ODE state deviating from GT trajectory
    if sigma_max > 0.0:
        log_sigma = (
            torch.rand(B, device=device) * (math.log(sigma_max) - math.log(sigma_min))
            + math.log(sigma_min)
        )
        sigma = torch.exp(log_sigma)
        x_t_input = x_t + sigma[:, None, None, None] * torch.randn_like(x_t)
    else:
        x_t_input = x_t.clone()

    # Explicit missing-frame batches: zero out x_t so the model must reconstruct
    # velocity from ctx_first, ctx_last, and t alone
    if missing_frac > 0.0:
        n_missing = max(1, int(B * missing_frac))
        miss_idx  = torch.randperm(B, device=device)[:n_missing]
        x_t_input = x_t_input.clone()
        x_t_input[miss_idx] = 0.0

    # Central FD velocity; dt_norm = 1/(T-1)
    dt_norm  = 1.0 / (T - 1)
    v_target = (x_next - x_prev) / (2.0 * dt_norm)         # [B, 1, H, W]

    v_pred = model(x_t_input, t_norm, ctx_first, ctx_last)
    return F.mse_loss(v_pred, v_target)


def compute_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.detach(), target.detach()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def train_flow_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    sigma_max: float = 0.0,
    missing_frac: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = flow_matching_loss(model, batch, device, sigma_max=sigma_max, missing_frac=missing_frac)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_flow_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            B, T = targets.shape[:2]
            ctx_first = targets[:, 0]
            ctx_last  = targets[:, -1]

            t_idx  = torch.randint(1, T - 1, (B,), device=device)
            t_norm = t_idx.float() / (T - 1)
            x_t    = targets[torch.arange(B), t_idx]
            x_prev = targets[torch.arange(B), t_idx - 1]
            x_next = targets[torch.arange(B), t_idx + 1]

            dt_norm  = 1.0 / (T - 1)
            v_target = (x_next - x_prev) / (2.0 * dt_norm)

            v_pred = model(x_t, t_norm, ctx_first, ctx_last)
            loss   = F.mse_loss(v_pred, v_target)

            # PSNR on the predicted next frame (one Euler step ahead)
            x_next_pred = x_t + (1.0 / (T - 1)) * v_pred
            total_psnr += compute_psnr(x_next_pred.clamp(0, 1), x_next)
            total_loss += loss.item()

    n = len(loader)
    return total_loss / n, total_psnr / n
