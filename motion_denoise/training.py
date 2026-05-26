import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

_LOG_SIGMA_MIN = math.log(0.01)
_LOG_SIGMA_MAX = math.log(1.5)


def dsm_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
) -> Tensor:
    targets = batch["targets"].to(device)   # [B, T, 1, H, W]
    B, T = targets.shape[:2]
    assert T >= 3, f"T must be >= 3, got {T}"

    ctx_first = targets[:, 0]              # [B, 1, H, W]
    ctx_last  = targets[:, -1]             # [B, 1, H, W]

    t_idx   = torch.randint(1, T - 1, (B,), device=device)
    x_clean = targets[torch.arange(B, device=device), t_idx]   # [B, 1, H, W]

    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    log_sigma = torch.rand(B, device=device) * (log_sigma_max - log_sigma_min) + log_sigma_min
    sigma     = torch.exp(log_sigma)                            # [B]

    x_noisy = x_clean + sigma[:, None, None, None] * torch.randn_like(x_clean)
    x_pred  = model(x_noisy, sigma, ctx_first, ctx_last)

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
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = dsm_loss(model, batch, device, sigma_min, sigma_max)
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

            log_sigma_min = math.log(sigma_min)
            log_sigma_max = math.log(sigma_max)
            log_sigma = (
                torch.rand(B, device=device) * (log_sigma_max - log_sigma_min) + log_sigma_min
            )
            sigma   = torch.exp(log_sigma)
            x_noisy = x_clean + sigma[:, None, None, None] * torch.randn_like(x_clean)
            x_pred  = model(x_noisy, sigma, ctx_first, ctx_last)

            total_loss += F.mse_loss(x_pred, x_clean).item()
            total_psnr += compute_psnr(x_pred, x_clean)

    n = len(loader)
    return total_loss / n, total_psnr / n
