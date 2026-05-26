"""
ODE integration for trajectory reconstruction using a trained FlowNet.

reconstruct_flow_trajectory
----------------------------
Integrates  dx/dt = v_θ(x, t; ctx_first, ctx_last)  from t=0 to t=1
using the Heun (trapezoidal predictor-corrector) method for accuracy with a
modest number of steps.

Returns the full sequence [T, 1, H, W] with x[0]=ctx_first, x[T-1]=ctx_last,
and all intermediate frames obtained by sampling the ODE trajectory at the
times  t_k = k / (T-1)  for k=1..T-2.

The boundary frames are pinned: regardless of numerical error the returned
tensor always begins with ctx_first and ends with ctx_last.
"""

import torch
from torch import Tensor


@torch.no_grad()
def reconstruct_flow_trajectory(
    model,
    ctx_first: Tensor,
    ctx_last:  Tensor,
    T:         int,
    device:    torch.device | None = None,
    n_steps:   int = 64,
) -> Tensor:
    """
    Parameters
    ----------
    model     : trained FlowNet
    ctx_first : [1, H, W] or [B, 1, H, W]  first clean frame
    ctx_last  : [1, H, W] or [B, 1, H, W]  last  clean frame
    T         : total number of frames (including endpoints)
    device    : inference device (defaults to ctx_first.device)
    n_steps   : number of Heun integration steps (more → more accurate)

    Returns
    -------
    frames : [T, 1, H, W] float32 tensor, values in [0, 1]
    """
    if device is None:
        device = ctx_first.device

    model.eval()

    # Ensure [1, 1, H, W] layout
    if ctx_first.dim() == 3:
        ctx_first = ctx_first.unsqueeze(0)
    if ctx_last.dim() == 3:
        ctx_last = ctx_last.unsqueeze(0)

    ctx_first = ctx_first.to(device)
    ctx_last  = ctx_last.to(device)

    dt = 1.0 / n_steps
    frame_times = [k / (T - 1) for k in range(1, T - 1)]   # times to record

    frames = [ctx_first[0].cpu()]   # t=0 pinned

    x = ctx_first.clone()
    t_curr = 0.0
    fi = 0   # next frame_times index to record

    for _ in range(n_steps):
        t0 = torch.tensor([t_curr],        dtype=torch.float32, device=device)
        t1 = torch.tensor([t_curr + dt],   dtype=torch.float32, device=device)

        # Heun predictor step
        v0 = model(x, t0, ctx_first, ctx_last)
        x_pred = x + dt * v0

        # Heun corrector step
        v1 = model(x_pred, t1, ctx_first, ctx_last)
        x = x + dt * 0.5 * (v0 + v1)

        t_curr += dt

        # Record any frame whose time falls within this step
        while fi < len(frame_times) and frame_times[fi] <= t_curr + dt * 0.5:
            frames.append(x[0].clamp(0, 1).cpu())
            fi += 1

    frames.append(ctx_last[0].cpu())   # t=1 pinned

    return torch.stack(frames, dim=0)  # [T, 1, H, W]
