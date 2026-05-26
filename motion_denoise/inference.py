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
    ctx_first / ctx_last.
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
            y = frames[t:t+1]                                    # [1, 1, H, W]
            sigma_val = noise_std if mask[t].item() else missing_sigma
            sigma_t   = torch.tensor([sigma_val], dtype=torch.float32, device=device)
            result[t] = model(y, sigma_t, ctx_first, ctx_last)[0]  # [1, H, W]

    return result.cpu()
