import torch
from torch import Tensor


def map_denoise_sequence(
    model,
    frames: Tensor,         # [T, 1, H, W]
    mask: Tensor,           # [T] bool, True=observed
    device: torch.device | None = None,
    noise_std: float = 0.15,
    missing_sigma: float = 0.8,
) -> Tensor:                # [T, 1, H, W] denoised
    """
    Single-step MAP denoising for all intermediate frames.

    Noisy observed frames: denoised at sigma=noise_std.
    Missing frames (mask=False): reconstructed from context at sigma=missing_sigma,
    which is large enough that the model ignores the zeroed input and relies on
    ctx_first / ctx_last, guided by the temporal position t_frac.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    frames = frames.to(device)
    T = frames.shape[0]
    ctx_first = frames[0:1]    # [1, 1, H, W]
    ctx_last  = frames[-1:]    # [1, 1, H, W]

    result = frames.clone()
    with torch.no_grad():
        for t in range(1, T - 1):
            y         = frames[t:t+1]
            sigma_val = noise_std if mask[t].item() else missing_sigma
            sigma_t   = torch.tensor([sigma_val], dtype=torch.float32, device=device)
            t_frac    = torch.tensor([t / (T - 1)], dtype=torch.float32, device=device)
            result[t] = model(y, sigma_t, ctx_first, ctx_last, t_frac)[0]

    return result.cpu()


def reconstruct_trajectory(
    model,
    ctx_first: Tensor,      # [1, H, W] or [1, 1, H, W]
    ctx_last: Tensor,       # [1, H, W] or [1, 1, H, W]
    T: int,
    device: torch.device | None = None,
    sigma: float = 0.8,
) -> Tensor:                # [T, 1, H, W]
    """
    Reconstruct all T-2 intermediate frames from only the first and last frames.

    No intermediate observations are used. The model uses temporal position
    conditioning (t_frac) to place each reconstructed frame correctly along
    the trajectory.
    """
    if device is None:
        device = next(model.parameters()).device

    if ctx_first.dim() == 3:
        ctx_first = ctx_first.unsqueeze(0)   # [1, 1, H, W]
    if ctx_last.dim() == 3:
        ctx_last = ctx_last.unsqueeze(0)

    ctx_first = ctx_first.to(device)
    ctx_last  = ctx_last.to(device)
    H, W = ctx_first.shape[-2], ctx_first.shape[-1]

    frames = torch.zeros(T, 1, H, W, device=device)
    frames[0]  = ctx_first[0]
    frames[-1] = ctx_last[0]

    model.eval()
    with torch.no_grad():
        for t in range(1, T - 1):
            t_frac  = torch.tensor([t / (T - 1)], dtype=torch.float32, device=device)
            sigma_t = torch.tensor([sigma],        dtype=torch.float32, device=device)
            x_blank = torch.zeros(1, 1, H, W, device=device)   # no observation
            frames[t] = model(x_blank, sigma_t, ctx_first, ctx_last, t_frac)[0]

    return frames.cpu()
