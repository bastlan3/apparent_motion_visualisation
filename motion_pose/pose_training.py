"""
DSM training in 3D pose space with the same curriculum as motion_denoise,
plus an optional differentiable rendering loss that closes the gap between
3D pose accuracy and 2D pixel quality.

Total loss = DSM pose loss  +  λ_render · render loss

DSM pose loss:
  t_idx ~ Uniform{1..T-2}
  σ ~ log-Uniform[σ_min, σ_max]
  pose_noisy = pose_t + σ · ε,   ε ~ N(0, I_{12})
  loss = MSE(PoseNet(pose_noisy, σ, pose_0, pose_T, t/(T-1)), pose_t)

Render loss (when render_lambda > 0):
  Recover (pos_pred, R_pred) from pose_pred.
  Re-orthogonalise R_pred via Gram-Schmidt so the orientation is a valid rotation.
  Compute world_vertices = shape.vertices @ R_pred.T + pos_pred.
  Render with the differentiable soft-wireframe renderer.
  render_loss = MSE(rendered_image, target_clean_image)

  Gradients flow:
    render_loss → rendered pixel values → projected pixel coords
                → world vertices → (pos_pred, R_pred) → pose_pred → PoseNet weights

This is what closes the loop: the optimizer sees pixel-space error directly.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from motion_render.shapes import cube as make_cube
from .diff_renderer import diff_render

_CUBE = make_cube()   # shared, CPU-side shape data


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


def _gram_schmidt(R: Tensor) -> Tensor:
    """Differentiable re-orthogonalisation of a 3×3 matrix via Gram–Schmidt."""
    c0 = R[:, 0]
    c0 = c0 / (c0.norm() + 1e-8)
    c1 = R[:, 1] - (c1_dot := (R[:, 1] * c0).sum()) * c0
    c1 = c1 / (c1.norm() + 1e-8)
    c2 = torch.cross(c0, c1, dim=0)
    return torch.stack([c0, c1, c2], dim=1)


def _render_loss_for_sample(
    pose_pred:  Tensor,   # [12]
    target_img: Tensor,   # [1, H, W]
    camera_eye: Tensor,   # [3]
    fov_y:      float = 0.785,
) -> Tensor:
    """Render pose_pred and compute MSE against target_img (differentiable)."""
    H, W = target_img.shape[-2:]
    pos  = pose_pred[:3]
    R    = _gram_schmidt(pose_pred[3:].reshape(3, 3))
    verts = _CUBE.vertices.to(pose_pred.device) @ R.T + pos
    rendered = diff_render(verts, _CUBE.edges.to(pose_pred.device),
                           camera_eye.to(pose_pred.device), H, W, fov_y)
    return F.mse_loss(rendered, target_img[0])


def pose_dsm_loss(
    model,
    batch: dict,
    device: torch.device,
    sigma_min: float = 0.01,
    sigma_max: float = 1.5,
    missing_frac: float = 0.0,
    missing_sigma: float = 0.8,
    render_lambda: float = 0.0,
) -> Tensor:
    poses      = batch["poses"].to(device)     # [B, T, 12]
    targets    = batch["targets"].to(device)   # [B, T, 1, H, W]
    camera_eyes = batch["camera_eye"].to(device) # [B, 3]
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
    dsm = F.mse_loss(pose_pred, pose_t)

    if render_lambda <= 0.0:
        return dsm

    # Differentiable rendering loss: pixel error flows back to PoseNet
    rl = torch.stack([
        _render_loss_for_sample(
            pose_pred[b],
            targets[b, t_idx[b]],
            camera_eyes[b],
        )
        for b in range(B)
    ]).mean()
    return dsm + render_lambda * rl


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
    render_lambda: float = 0.0,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = pose_dsm_loss(model, batch, device,
                             sigma_max=sigma_max,
                             missing_frac=missing_frac,
                             missing_sigma=missing_sigma,
                             render_lambda=render_lambda)
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
