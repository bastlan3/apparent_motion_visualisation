"""
Differentiable wireframe renderer using soft Gaussian edge blobs.

Each edge is rendered as a soft "tube" in image space: every pixel receives
exp(-d²/2σ²) where d is the pixel's distance to the nearest point on the
projected line segment.  The final image is the element-wise maximum over all
edges.

Because the projected pixel coordinates are computed with standard PyTorch
matrix operations, gradients flow back through the rendered image all the way
to the 3D world vertices — and from there to the pose (position + orientation)
that placed those vertices.

This provides the missing supervision signal for PoseNet:
    ∂(pixel_loss) / ∂(pose)  through  rendering → projection → pose
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from motion_render.transforms import look_at_matrix


def _project_differentiable(
    world_pts: Tensor,   # [N, 3]  requires_grad OK
    eye:       Tensor,   # [3]
    fov_y:     float,
    H:         int,
    W:         int,
) -> Tensor:
    """
    Differentiable perspective projection.
    Returns pixel coordinates [N, 2]; gradients flow to world_pts.
    """
    up  = torch.tensor([0.0, 1.0, 0.0], device=world_pts.device, dtype=torch.float32)
    eye = eye.float().to(world_pts.device)
    V   = look_at_matrix(eye, torch.zeros(3, device=world_pts.device), up)  # [4, 4]

    N    = world_pts.shape[0]
    ones = torch.ones(N, 1, dtype=torch.float32, device=world_pts.device)
    pts_h   = torch.cat([world_pts.float(), ones], dim=1)           # [N, 4]
    pts_cam = (V @ pts_h.T).T                                       # [N, 4]

    z = pts_cam[:, 2]                                               # [N]
    # Avoid divide-by-zero; behind-camera points will produce off-screen pixels
    z_safe = torch.where(z < -1e-3, z, torch.full_like(z, -1e-3))

    f      = 1.0 / math.tan(fov_y / 2.0)
    aspect = W / H
    x_ndc  = pts_cam[:, 0] * f / (aspect * (-z_safe))
    y_ndc  = pts_cam[:, 1] * f / (-z_safe)

    px_x = (x_ndc + 1.0) * 0.5 * W
    px_y = (1.0 - y_ndc) * 0.5 * H
    return torch.stack([px_x, px_y], dim=1)                        # [N, 2]


def diff_render(
    world_vertices: Tensor,   # [N, 3]  differentiable
    edges:          Tensor,   # [E, 2]  long, fixed topology
    eye:            Tensor,   # [3]
    H:              int,
    W:              int,
    fov_y:          float = 0.785,
    sigma:          float = 0.8,   # edge softness in pixels
) -> Tensor:
    """
    Differentiable soft-wireframe rendering.
    Returns [H, W] float32 image in approximately [0, 1].
    Gradients flow to world_vertices (and hence to pose).
    """
    px = _project_differentiable(world_vertices, eye, fov_y, H, W)  # [N, 2]

    # Build pixel coordinate grid  [H, W, 2]
    ys  = torch.arange(H, dtype=torch.float32, device=px.device)
    xs  = torch.arange(W, dtype=torch.float32, device=px.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1)                            # [H, W, 2]

    # Edge endpoints [E, 2]
    p0 = px[edges[:, 0]]   # [E, 2]
    p1 = px[edges[:, 1]]   # [E, 2]

    # Broadcast: grid [H, W, 1, 2], edges [1, 1, E, 2]
    g  = grid.unsqueeze(2)                  # [H, W, 1, 2]
    q0 = p0.unsqueeze(0).unsqueeze(0)       # [1, 1, E, 2]
    q1 = p1.unsqueeze(0).unsqueeze(0)       # [1, 1, E, 2]

    v  = q1 - q0                            # [1, 1, E, 2]  edge direction
    w  = g  - q0                            # [H, W, E, 2]  pixel relative to p0

    c1 = (w * v).sum(-1)                    # [H, W, E]  projection onto edge
    c2 = (v * v).sum(-1) + 1e-8            # [1, 1, E]  edge length²
    t  = (c1 / c2).clamp(0.0, 1.0)         # [H, W, E]  clamp to segment

    closest  = q0 + t.unsqueeze(-1) * v    # [H, W, E, 2]
    dist_sq  = ((g - closest) ** 2).sum(-1) # [H, W, E]

    influence = torch.exp(-dist_sq / (2.0 * sigma ** 2))   # [H, W, E]
    return influence.max(dim=2).values                       # [H, W]
