import random
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .scene import Scene
from .camera import Camera
from .renderer import Renderer


class MotionDataset(Dataset):
    def __init__(
        self,
        n_sequences: int = 1000,
        T: int = 16,
        dt: float = 0.05,
        image_height: int = 64,
        image_width: int = 64,
        fov_y: float = 0.785,       # ~45 degrees
        camera_radius: float = 5.0,
        n_bodies: int = 2,
        motion_modes: tuple = ('translate', 'rotate', 'both'),
        pivot_spread: float = 0.0,
        seed: int = None,
    ):
        self.n_sequences = n_sequences
        self.T = T
        self.dt = dt
        self.image_height = image_height
        self.image_width = image_width
        self.fov_y = fov_y
        self.camera_radius = camera_radius
        self.n_bodies = n_bodies
        self.motion_modes = motion_modes
        self.pivot_spread = pivot_spread
        self.seed = seed

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tensor:
        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        scene = Scene.random(
            n_bodies=self.n_bodies,
            motion_modes=self.motion_modes,
            pivot_spread=self.pivot_spread,
        )
        camera = Camera(self.image_height, self.image_width, self.fov_y)
        camera.place_random(self.camera_radius)
        renderer = Renderer(self.image_height, self.image_width)

        frames = []
        for _ in range(self.T):
            frame = renderer.render_scene(scene, camera)
            frames.append(frame)
            scene.step(self.dt)

        return torch.stack(frames, dim=0)  # [T, H, W]
