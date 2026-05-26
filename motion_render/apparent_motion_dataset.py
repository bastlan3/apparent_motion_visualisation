import random
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .scene import Scene
from .camera import Camera
from .renderer import Renderer

CorruptionMode = Literal["none", "noisy", "missing", "mixed", "all_missing"]

_SHAPE_TYPES = ("cube", "sphere", "tetrahedron")
_MOTION_MODES = ("translate", "rotate", "both")


class ApparentMotionDataset(Dataset):
    """
    Generates single-body apparent motion sequences.

    Each item contains a clean sequence and a corrupted version where
    intermediate frames may be noised or zeroed out. First and last frames
    are always clean.

    Returns dicts with:
      "frames":  FloatTensor [T, 1, H, W]  - corrupted input (values in [0, 1])
      "targets": FloatTensor [T, 1, H, W]  - clean ground truth
      "mask":    BoolTensor  [T]           - True=observed, False=zeroed/missing
    """

    def __init__(
        self,
        *,
        n_sequences: int = 1000,
        T: int = 16,
        dt: float = 0.05,
        image_height: int = 64,
        image_width: int = 64,
        fov_y: float = 0.785,
        camera_radius: float = 5.0,
        shape_type: str | None = None,
        motion_mode: str | None = None,
        corruption_mode: CorruptionMode = "mixed",
        noise_std: float = 0.15,
        missing_prob: float = 0.3,
        seed: int | None = 42,
        trans_speed_range: tuple | None = None,
        rot_speed_range: tuple | None = None,
    ) -> None:
        if T < 3:
            raise ValueError(f"T must be >= 3, got {T}")
        if shape_type is not None and shape_type not in _SHAPE_TYPES:
            raise ValueError(f"shape_type must be one of {_SHAPE_TYPES} or None")
        if motion_mode is not None and motion_mode not in _MOTION_MODES:
            raise ValueError(f"motion_mode must be one of {_MOTION_MODES} or None")

        self.n_sequences = n_sequences
        self.T = T
        self.dt = dt
        self.image_height = image_height
        self.image_width = image_width
        self.fov_y = fov_y
        self.camera_radius = camera_radius
        self.shape_types = (shape_type,) if shape_type is not None else _SHAPE_TYPES
        self.motion_modes = (motion_mode,) if motion_mode is not None else _MOTION_MODES
        self.corruption_mode = corruption_mode
        self.noise_std = noise_std
        self.missing_prob = missing_prob
        self.seed = seed
        self.trans_speed_range = trans_speed_range
        self.rot_speed_range = rot_speed_range

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> dict:
        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)

        scene = Scene.random(
            n_bodies=1, shape_types=self.shape_types, motion_modes=self.motion_modes,
            trans_speed_range=self.trans_speed_range, rot_speed_range=self.rot_speed_range,
        )
        camera = Camera(self.image_height, self.image_width, self.fov_y)
        camera.place_random(self.camera_radius)
        renderer = Renderer(self.image_height, self.image_width)

        # Render clean sequence
        clean_frames = []
        for _ in range(self.T):
            frame = renderer.render_scene(scene, camera).unsqueeze(0)  # [1, H, W]
            clean_frames.append(frame)
            scene.step(self.dt)

        targets = torch.stack(clean_frames, dim=0)   # [T, 1, H, W]
        frames = targets.clone()
        mask = torch.ones(self.T, dtype=torch.bool)

        # Corrupt intermediate frames only
        if self.corruption_mode == "all_missing":
            frames[1:-1] = 0.0
            mask[1:-1] = False
        else:
            for t in range(1, self.T - 1):
                decision = self._corruption_decision()
                if decision == "missing":
                    frames[t] = 0.0
                    mask[t] = False
                elif decision == "noisy":
                    noise = torch.randn_like(frames[t]) * self.noise_std
                    frames[t] = (frames[t] + noise).clamp(0.0, 1.0)
                    # mask stays True — frame is present but degraded

        return {"frames": frames, "targets": targets, "mask": mask}

    def _corruption_decision(self) -> str:
        if self.corruption_mode == "none":
            return "clean"
        if self.corruption_mode == "noisy":
            return "noisy"
        if self.corruption_mode == "missing":
            return "missing"
        # mixed: missing with probability missing_prob, else noisy
        return "missing" if random.random() < self.missing_prob else "noisy"


# ── Collation helpers ─────────────────────────────────────────────────────────

def sequence_collate_fn(batch: list[dict]) -> dict:
    """Standard UNet collation: tensors are [B, T, 1, H, W] / [B, T]."""
    return {
        "frames":  torch.stack([item["frames"]  for item in batch]),   # [B, T, 1, H, W]
        "targets": torch.stack([item["targets"] for item in batch]),   # [B, T, 1, H, W]
        "mask":    torch.stack([item["mask"]    for item in batch]),   # [B, T]
    }


def rnn_collate_fn(batch: list[dict]) -> dict:
    """RNN collation: time-first layout [T, B, 1, H, W] / [T, B]."""
    frames  = torch.stack([item["frames"]  for item in batch])   # [B, T, 1, H, W]
    targets = torch.stack([item["targets"] for item in batch])   # [B, T, 1, H, W]
    mask    = torch.stack([item["mask"]    for item in batch])   # [B, T]
    return {
        "frames":  frames.transpose(0, 1),    # [T, B, 1, H, W]
        "targets": targets.transpose(0, 1),   # [T, B, 1, H, W]
        "mask":    mask.t(),                  # [T, B]
    }
