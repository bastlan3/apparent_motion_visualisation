"""
Equilibrium matching inference for 3D pose trajectories.

reconstruct_pose_trajectory
---------------------------
Uses annealed Langevin dynamics in 12D pose space to sample from

    p(pose_1, ..., pose_{T-2} | pose_0, pose_{T-1})

The Langevin chain is initialised from the linear interpolant and driven
by the learned score  ∇_x log p(x | pose_0, pose_T, t).  After annealing
through a sequence of decreasing σ values the chain reaches its stationary
("equilibrium") distribution, giving physically plausible intermediate poses.

render_pose_sequence
--------------------
Renders a sequence of raw poses to 2D images using the existing wireframe
renderer.  The camera eye position saved by the dataset is used so the
viewpoint matches what was used to generate the training frames.
"""

import math
import torch
from torch import Tensor

from motion_render.body import RigidBody
from motion_render.scene import Scene
from motion_render.camera import Camera
from motion_render.renderer import Renderer
from motion_render.shapes import cube as make_cube


# ── annealed Langevin sampling ─────────────────────────────────────────────────

SIGMA_SCHEDULE = [1.2, 0.6, 0.3, 0.15, 0.07, 0.03]   # coarse→fine
N_STEPS        = 20    # Langevin steps per σ level
STEP_COEFF     = 0.01  # Langevin step size = STEP_COEFF * σ²


@torch.no_grad()
def reconstruct_pose_trajectory(
    model,
    pose_0: Tensor,   # [12]  raw world coords
    pose_T: Tensor,   # [12]  raw world coords
    T:      int,
    device: torch.device | None = None,
) -> Tensor:
    """
    Returns [T, 12]  raw pose tensor.
    First and last rows are the pinned endpoint poses.
    Intermediate rows are sampled via annealed Langevin dynamics.
    """
    if device is None:
        device = pose_0.device

    model.eval()
    pose_0 = pose_0.to(device)
    pose_T = pose_T.to(device)

    n_inter = T - 2   # number of intermediate frames
    t_fracs = torch.tensor(
        [(k + 1) / (T - 1) for k in range(n_inter)],
        dtype=torch.float32, device=device,
    )  # [n_inter]

    # Expand endpoint poses to batch over all intermediate frames at once
    p0_batch = pose_0.unsqueeze(0).expand(n_inter, -1)   # [n_inter, 12]
    pT_batch = pose_T.unsqueeze(0).expand(n_inter, -1)   # [n_inter, 12]

    # Initialise from straight-line interpolant in pose space
    x = torch.stack([
        (1 - tf) * pose_0 + tf * pose_T for tf in t_fracs
    ])  # [n_inter, 12]

    # Annealed Langevin dynamics
    for sigma in SIGMA_SCHEDULE:
        sigma_batch = torch.full((n_inter,), sigma, dtype=torch.float32, device=device)
        step_size   = STEP_COEFF * sigma ** 2

        for _ in range(N_STEPS):
            x_pred  = model(x, sigma_batch, p0_batch, pT_batch, t_fracs)
            score   = (x_pred - x) / (sigma ** 2)
            noise   = torch.randn_like(x) * math.sqrt(2.0 * step_size)
            x       = x + step_size * score + noise

    # Assemble full trajectory
    all_poses = torch.zeros(T, 12, device=device)
    all_poses[0]    = pose_0
    all_poses[-1]   = pose_T
    all_poses[1:-1] = x
    return all_poses.cpu()


# ── rendering ──────────────────────────────────────────────────────────────────

def render_pose_sequence(
    poses:      Tensor,      # [T, 12]  raw pose tensor
    camera_eye: Tensor,      # [3]  camera position
    image_height: int = 64,
    image_width:  int = 64,
    fov_y: float = 0.785,
    camera_radius: float = 5.0,
) -> Tensor:
    """
    Renders each pose as a 2D wireframe image.
    Returns [T, 1, H, W] float32 tensor in [0, 1].
    """
    shape    = make_cube()
    camera   = Camera(image_height, image_width, fov_y)
    camera.eye = camera_eye.float()
    renderer = Renderer(image_height, image_width)

    frames = []
    for pose_vec in poses:
        pos = pose_vec[:3].float()
        rot = pose_vec[3:].reshape(3, 3).float()

        # Re-orthogonalise via QR to correct for Langevin drift
        Q, _ = torch.linalg.qr(rot)

        body  = RigidBody(
            shape=shape, position=pos, orientation=Q, mode="rotate",
            velocity=torch.zeros(3), rot_axis=torch.tensor([0.0, 0.0, 1.0]),
            rot_speed=0.0,
        )
        scene = Scene([body])
        frame = renderer.render_scene(scene, camera).unsqueeze(0)  # [1, H, W]
        frames.append(frame)

    return torch.stack(frames, dim=0)   # [T, 1, H, W]


def pose_to_2d_sequence(
    poses:      Tensor,   # [T, 12]
    camera_eye: Tensor,   # [3]
    targets:    Tensor,   # [T, 1, H, W]  for shape
) -> Tensor:
    """Convenience wrapper: renders poses and pins endpoints to clean GT frames."""
    H, W   = targets.shape[-2:]
    frames = render_pose_sequence(poses, camera_eye, H, W)
    frames[0]  = targets[0]
    frames[-1] = targets[-1]
    return frames
