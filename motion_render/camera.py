import math
import torch
from torch import Tensor

from .transforms import look_at_matrix


class Camera:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        fov_y: float = math.pi / 4,
        near: float = 0.1,
        far: float = 50.0,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.fov_y = fov_y
        self.near = near
        self.far = far
        self.eye = torch.tensor([0.0, 0.0, 5.0])
        self.target = torch.zeros(3)

    def place_random(self, radius: float = 5.0) -> None:
        v = torch.randn(3)
        self.eye = radius * v / v.norm()

    def project(self, world_points: Tensor):
        """
        world_points: [N, 3]
        Returns (px [N,2] float32, valid [N] bool).
        px contains pixel coordinates; valid marks points in front of the camera.
        """
        N = world_points.shape[0]
        up = torch.tensor([0.0, 1.0, 0.0])
        V = look_at_matrix(self.eye, self.target, up)

        ones = torch.ones(N, 1, dtype=torch.float32)
        pts_h = torch.cat([world_points.float(), ones], dim=1)  # [N, 4]
        pts_cam = (V @ pts_h.T).T                               # [N, 4]

        z_cam = pts_cam[:, 2]
        valid = z_cam < -self.near

        z_safe = torch.where(valid, z_cam, torch.full_like(z_cam, -(self.near + 1e-6)))

        f = 1.0 / math.tan(self.fov_y / 2.0)
        aspect = self.image_width / self.image_height

        x_ndc = pts_cam[:, 0] * f / (aspect * (-z_safe))
        y_ndc = pts_cam[:, 1] * f / (-z_safe)

        px_x = (x_ndc + 1.0) * 0.5 * self.image_width
        px_y = (1.0 - y_ndc) * 0.5 * self.image_height
        px = torch.stack([px_x, px_y], dim=1)

        return px, valid
