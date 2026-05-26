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
        pivot_point: Tensor | None = None,
    ):
        self.shape = shape
        self.position = position.float()
        self.orientation = orientation.float()
        self.mode = mode
        self.velocity = velocity.float()
        self.rot_axis = rot_axis.float()
        self.rot_speed = rot_speed
        # pivot_point is used by 'orbit' and 'orbit_spin' modes only
        self.pivot_point = pivot_point.float() if pivot_point is not None else None
        self._step_count = 0

    def step(self, dt: float) -> None:
        if self.mode in ('translate', 'both'):
            self.position = self.position + self.velocity * dt

        if self.mode in ('rotate', 'both'):
            delta_R = rotation_matrix(self.rot_axis, self.rot_speed * dt)
            self.orientation = delta_R @ self.orientation

        if self.mode in ('orbit', 'orbit_spin'):
            # Revolve position around pivot_point (off-centre rotation)
            delta_R = rotation_matrix(self.rot_axis, self.rot_speed * dt)
            rel = self.position - self.pivot_point
            self.position = self.pivot_point + delta_R @ rel
            if self.mode == 'orbit_spin':
                # Also spin the body around its own axis while orbiting
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
        trans_speed_range: tuple | None = None,
        rot_speed_range: tuple | None = None,
    ) -> 'RigidBody':
        # trans/rot_speed_range override speed_range when provided
        tlo, thi = trans_speed_range if trans_speed_range is not None else speed_range
        rlo, rhi = rot_speed_range   if rot_speed_range   is not None else speed_range

        orientation = torch.eye(3, dtype=torch.float32)
        rot_axis = torch.randn(3)
        rot_axis = rot_axis / (rot_axis.norm() + 1e-8)
        rot_speed = rlo + random.random() * (rhi - rlo)

        pivot_point = None
        if mode in ('orbit', 'orbit_spin'):
            # Pivot somewhere near the scene centre; body starts at a random
            # orbital radius (0.4–1.0) away from the pivot.
            pivot_point = (torch.rand(3) * 2 - 1) * (position_range * 0.5)
            orbit_r = 0.4 + random.random() * 0.6
            offset = torch.randn(3)
            offset = offset / (offset.norm() + 1e-8) * orbit_r
            position = pivot_point + offset
        else:
            position = (torch.rand(3) * 2 - 1) * position_range

        speed = tlo + random.random() * (thi - tlo)
        v = torch.randn(3)
        velocity = v / (v.norm() + 1e-8) * speed

        return RigidBody(
            shape=shape,
            position=position,
            orientation=orientation,
            mode=mode,
            velocity=velocity,
            rot_axis=rot_axis,
            rot_speed=rot_speed,
            pivot_point=pivot_point,
        )
