"""
Equilibrium matching inference for spatiotemporal volumes.

reconstruct_st_trajectory
--------------------------
Uses annealed Langevin dynamics on the full (T-2) × H × W intermediate volume to
sample from

    p(frame_1, ..., frame_{T-2} | frame_0, frame_{T-1})

Langevin dynamics in the full spatiotemporal volume gives the model access to
correlations across both space and time simultaneously — it is not independent
per-frame inference.

After annealing, the reconstructed volume is assembled with the pinned endpoints
into a [T, 1, H, W] sequence.

st_velocity_field
-----------------
Computes an approximate optical-flow-style velocity field from the reconstructed
sequence using finite differences in space and time:

    I_x(x, y, t) = ∂I/∂x   (central FD in x)
    I_y(x, y, t) = ∂I/∂y   (central FD in y)
    I_t(x, y, t) = ∂I/∂t   (central FD in t)

At each (x, y, t) the brightness-constancy equation gives the spatial motion
direction orthogonal to edges; we return the (I_x, I_y, I_t) gradient vector for
direct visualisation as a spatiotemporal vector field.
"""

import math
import torch
from torch import Tensor


SIGMA_SCHEDULE = [1.2, 0.8, 0.5, 0.25, 0.12, 0.05, 0.02]
N_STEPS        = 20
STEP_COEFF     = 0.01


@torch.no_grad()
def reconstruct_st_trajectory(
    model,
    ctx_first: Tensor,              # [1, 1, H, W] or [1, H, W]
    ctx_last:  Tensor,              # [1, 1, H, W] or [1, H, W]
    T:         int,
    device:    torch.device | None = None,
) -> Tensor:
    """
    Returns [T, 1, H, W]  full reconstructed sequence (CPU).
    Endpoints are pinned to the supplied ctx tensors.
    """
    if device is None:
        device = ctx_first.device

    ctx_f = ctx_first.to(device)
    ctx_l = ctx_last.to(device)

    # Ensure [1, 1, H, W]
    if ctx_f.dim() == 3:
        ctx_f = ctx_f.unsqueeze(0)
        ctx_l = ctx_l.unsqueeze(0)

    H, W = ctx_f.shape[-2:]
    Tm   = T - 2

    # Initialise from linear interpolation in pixel space
    ts = torch.linspace(1 / (T - 1), (T - 2) / (T - 1), Tm, device=device)  # [Tm]
    vol = (
        (1 - ts)[None, None, :, None, None] * ctx_f.unsqueeze(2)
        + ts[None, None, :, None, None]      * ctx_l.unsqueeze(2)
    )  # [1, 1, Tm, H, W]
    vol = vol + 0.05 * torch.randn_like(vol)

    # Broadcast ctx for model conditioning: [1, 1, Tm, H, W]
    ctx_f_b = ctx_f.unsqueeze(2).expand(-1, -1, Tm, -1, -1)
    ctx_l_b = ctx_l.unsqueeze(2).expand(-1, -1, Tm, -1, -1)

    model.eval()
    for sig in SIGMA_SCHEDULE:
        sigma_t   = torch.tensor([sig], dtype=torch.float32, device=device)
        step_size = STEP_COEFF * sig ** 2
        for _ in range(N_STEPS):
            vol_pred = model(vol, sigma_t, ctx_f_b, ctx_l_b)           # [1, 1, Tm, H, W]
            score    = (vol_pred - vol) / (sig ** 2)
            noise    = torch.randn_like(vol)
            vol      = vol + step_size * score + math.sqrt(2 * step_size) * noise

    # Assemble [T, 1, H, W] on CPU
    inter = vol[0].permute(1, 0, 2, 3).cpu()  # [Tm, 1, H, W]
    result = torch.cat([ctx_f[0:1].cpu(), inter, ctx_l[0:1].cpu()], dim=0)
    return result   # [T, 1, H, W]


def st_velocity_field(
    seq: Tensor,      # [T, 1, H, W]
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute spatiotemporal gradient field via central finite differences.

    Returns:
        I_x : [T-2, H-2, W-2]  ∂I/∂x
        I_y : [T-2, H-2, W-2]  ∂I/∂y
        I_t : [T-2, H-2, W-2]  ∂I/∂t

    Interior points only (edge pixels removed to avoid boundary artefacts).
    """
    I = seq[:, 0]   # [T, H, W]
    T, H, W = I.shape

    I_x = (I[:, 1:-1, 2:]  - I[:, 1:-1, :-2]) * 0.5   # [T, H-2, W-2]
    I_y = (I[:, 2:,   1:-1] - I[:, :-2, 1:-1]) * 0.5  # [T, H-2, W-2]
    I_t = (I[2:,  1:-1, 1:-1] - I[:-2, 1:-1, 1:-1]) * 0.5  # [T-2, H-2, W-2]

    # Trim T for I_x, I_y to match I_t
    I_x = I_x[1:-1]   # [T-2, H-2, W-2]
    I_y = I_y[1:-1]

    return I_x, I_y, I_t
