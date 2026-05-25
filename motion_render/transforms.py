import math
import torch
from torch import Tensor


def rotation_matrix(axis: Tensor, angle: float) -> Tensor:
    """Rodrigues' rotation formula. axis: [3], angle: radians → R: [3,3]"""
    k = axis.float()
    k = k / (k.norm() + 1e-8)
    kx, ky, kz = k[0].item(), k[1].item(), k[2].item()
    c = math.cos(angle)
    s = math.sin(angle)

    K = torch.tensor([
        [  0.0, -kz,   ky],
        [ kz,    0.0, -kx],
        [-ky,   kx,    0.0],
    ], dtype=torch.float32)

    I = torch.eye(3, dtype=torch.float32)
    outer = k.unsqueeze(1) * k.unsqueeze(0)

    return c * I + s * K + (1.0 - c) * outer


def look_at_matrix(eye: Tensor, target: Tensor, up: Tensor) -> Tensor:
    """Build a 4×4 view matrix (world → camera). Camera looks down -Z."""
    forward = target.float() - eye.float()
    forward = forward / (forward.norm() + 1e-8)

    world_up = up.float()
    if abs(torch.dot(forward, world_up).item()) > 0.99:
        world_up = torch.tensor([1.0, 0.0, 0.0])

    right = torch.cross(forward, world_up, dim=0)
    right = right / (right.norm() + 1e-8)
    true_up = torch.cross(right, forward, dim=0)

    eye_f = eye.float()
    M = torch.eye(4, dtype=torch.float32)
    M[0, :3] = right
    M[1, :3] = true_up
    M[2, :3] = -forward
    M[0, 3] = -torch.dot(right, eye_f)
    M[1, 3] = -torch.dot(true_up, eye_f)
    M[2, 3] =  torch.dot(forward, eye_f)

    return M


def apply_transform(vertices: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """vertices: [N,3], R: [3,3], t: [3] → [N,3]"""
    return vertices @ R.T + t
