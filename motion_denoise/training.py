import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

_LOG_SIGMA_MIN = math.log(0.01)
_LOG_SIGMA_MAX = math.log(1.5)


def sigma_max_schedule(
    epoch: int,
    total_epochs: int,
    sigma_start: float = 0.15,
    sigma_target: float = 1.5,
    schedule: str = "linear",
    warmup_frac: float = 0.67,
) -> float:
    """
    Returns σ_max for the current epoch.

    schedule:
      "linear"  – constant ramp
      "cosine"  – slow start, fast middle, slow end (smoother)
      "exp"     – log-linear; matches log-uniform sigma sampling
    warmup_frac – fraction of total_epochs over which the ramp runs; held
                  at sigma_target afterwards.
    """
    warmup = max(1, int(total_epochs * warmup_frac))
    t = min(epoch / warmup, 1.0)

    if schedule == "cosine":
        t = (1.0 - math.cos(math.pi * t)) / 2.0
    elif schedule == "exp":
        t = (math.exp(t) - 1.0) / (math.e - 1.0)
    # "linear": t unchanged

    return sigma_start + (sigma_target - sigma_start) * t


def dsm_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
    missing_frac: float = 0.0,
    missing_sigma: float = 0.8,
    noise_type: str = "gaussian",
) -> Tensor:
    """
    Denoising score matching loss.

    noise_type:
      "gaussian"    – x + σ·ε,  ε ~ N(0,I)
      "uniform"     – x + σ·ε,  ε ~ U(-√3, √3)  (same variance, bounded)
      "saltpepper"  – pixels randomly set to 0 or 1 with prob ∝ σ

    missing_frac: fraction of each batch trained as fully-zeroed frames
      (x_noisy=0, σ=missing_sigma), matching reconstruct_trajectory inference.
    """
    targets = batch["targets"].to(device)   # [B, T, 1, H, W]
    B, T = targets.shape[:2]
    assert T >= 3, f"T must be >= 3, got {T}"

    ctx_first = targets[:, 0]
    ctx_last  = targets[:, -1]

    t_idx   = torch.randint(1, T - 1, (B,), device=device)
    x_clean = targets[torch.arange(B, device=device), t_idx]
    t_frac  = t_idx.float() / (T - 1)

    log_sigma = (
        torch.rand(B, device=device) * (math.log(sigma_max) - math.log(sigma_min))
        + math.log(sigma_min)
    )
    sigma = torch.exp(log_sigma)

    # Corrupt according to noise_type
    if noise_type == "gaussian":
        x_noisy = x_clean + sigma[:, None, None, None] * torch.randn_like(x_clean)
    elif noise_type == "uniform":
        eps = (torch.rand_like(x_clean) * 2.0 - 1.0) * math.sqrt(3.0)
        x_noisy = x_clean + sigma[:, None, None, None] * eps
    elif noise_type == "saltpepper":
        prob = sigma[:, None, None, None].clamp(max=0.9)
        r = torch.rand_like(x_clean)
        x_noisy = x_clean.clone()
        x_noisy = torch.where(r < prob * 0.5,  torch.zeros_like(x_clean), x_noisy)
        x_noisy = torch.where((r >= prob * 0.5) & (r < prob), torch.ones_like(x_clean), x_noisy)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    # Override a fraction of the batch with explicitly zeroed frames
    if missing_frac > 0.0:
        n_missing = max(1, int(B * missing_frac))
        miss_idx  = torch.randperm(B, device=device)[:n_missing]
        x_noisy = x_noisy.clone()
        x_noisy[miss_idx] = 0.0
        sigma = sigma.clone()
        sigma[miss_idx] = missing_sigma

    x_pred = model(x_noisy, sigma, ctx_first, ctx_last, t_frac)
    return F.mse_loss(x_pred, x_clean)


def compute_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.detach(), target.detach()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
    grad_clip: float = 1.0,
    missing_frac: float = 0.0,
    missing_sigma: float = 0.8,
    noise_type: str = "gaussian",
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = dsm_loss(
            model, batch, device,
            sigma_min=sigma_min, sigma_max=sigma_max,
            missing_frac=missing_frac, missing_sigma=missing_sigma,
            noise_type=noise_type,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
) -> tuple[float, float]:
    # Always uses gaussian for fair cross-variant comparison
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            targets   = batch["targets"].to(device)
            B, T      = targets.shape[:2]
            ctx_first = targets[:, 0]
            ctx_last  = targets[:, -1]

            t_idx     = torch.randint(1, T - 1, (B,), device=device)
            x_clean   = targets[torch.arange(B, device=device), t_idx]
            t_frac    = t_idx.float() / (T - 1)

            log_sigma = (
                torch.rand(B, device=device) * (math.log(sigma_max) - math.log(sigma_min))
                + math.log(sigma_min)
            )
            sigma   = torch.exp(log_sigma)
            x_noisy = x_clean + sigma[:, None, None, None] * torch.randn_like(x_clean)
            x_pred  = model(x_noisy, sigma, ctx_first, ctx_last, t_frac)

            total_loss += F.mse_loss(x_pred, x_clean).item()
            total_psnr += compute_psnr(x_pred, x_clean)

    n = len(loader)
    return total_loss / n, total_psnr / n
