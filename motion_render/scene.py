import random

from .body import RigidBody
from .shapes import cube, sphere, tetrahedron

_SHAPE_FACTORIES = {
    'cube': cube,
    'sphere': sphere,
    'tetrahedron': tetrahedron,
}


class Scene:
    def __init__(self, bodies: list):
        self.bodies = bodies

    @staticmethod
    def random(
        n_bodies: int = 2,
        shape_types: tuple = ('cube', 'sphere', 'tetrahedron'),
        motion_modes: tuple = ('translate', 'rotate', 'both'),
        position_range: float = 1.5,
        trans_speed_range: tuple | None = None,
        rot_speed_range: tuple | None = None,
    ) -> 'Scene':
        bodies = []
        for _ in range(n_bodies):
            shape = _SHAPE_FACTORIES[random.choice(shape_types)]()
            mode = random.choice(motion_modes)
            bodies.append(RigidBody.random(
                shape, mode, position_range,
                trans_speed_range=trans_speed_range,
                rot_speed_range=rot_speed_range,
            ))
        return Scene(bodies)

    def step(self, dt: float = 0.05) -> None:
        for body in self.bodies:
            body.step(dt)
