import random
import torch
from torch import Tensor

from .shapes import ShapeData
from .transforms import rotation_matrix


class RigidBody:
    def __init__(
        self,
        shape: ShapeData,
        position: Tensor,
        orientation: Tensor,
        mode: str,
        velocity: Tensor,
        rot_axis: Tensor,
        rot_speed: float,
    ):
        self.shape = shape
        self.position = position.float()
        self.orientation = orientation.float()
        self.mode = mode
        self.velocity = velocity.float()
        self.rot_axis = rot_axis.float()
        self.rot_speed = rot_speed
        self._step_count = 0

    def step(self, dt: float) -> None:
        if self.mode in ('translate', 'both'):
            self.position = self.position + self.velocity * dt

        if self.mode in ('rotate', 'both'):
            delta_R = rotation_matrix(self.rot_axis, self.rot_speed * dt)
            self.orientation = delta_R @ self.orientation

        self._step_count += 1
        if self._step_count % 50 == 0:
            Q, _ = torch.linalg.qr(self.orientation)
            self.orientation = Q

    def world_vertices(self) -> Tensor:
        return self.shape.vertices @ self.orientation.T + self.position

    @staticmethod
    def random(
        shape: ShapeData,
        mode: str,
        position_range: float = 1.5,
        speed_range: tuple = (0.3, 1.5),
    ) -> 'RigidBody':
        lo, hi = speed_range

        position = (torch.rand(3) * 2 - 1) * position_range
        orientation = torch.eye(3, dtype=torch.float32)

        speed = lo + random.random() * (hi - lo)
        v = torch.randn(3)
        velocity = v / (v.norm() + 1e-8) * speed

        rot_axis = torch.randn(3)
        rot_axis = rot_axis / (rot_axis.norm() + 1e-8)
        rot_speed = lo + random.random() * (hi - lo)

        return RigidBody(
            shape=shape,
            position=position,
            orientation=orientation,
            mode=mode,
            velocity=velocity,
            rot_axis=rot_axis,
            rot_speed=rot_speed,
        )
