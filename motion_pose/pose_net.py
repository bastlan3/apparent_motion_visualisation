"""
PoseNet: score model in 3D pose space for equilibrium matching.

State vector
------------
pose = cat([position, orientation.flatten()])  ∈  R^12
  position    : 3D world coordinates
  orientation : 3×3 rotation matrix flattened to 9D

The model is conditioned on:
  - noise level σ  (for DSM training)
  - temporal position t ∈ [0,1]
  - endpoint poses  pose_0, pose_T

Internal normalisation (POSE_SCALE=3.0) keeps position components in ≈[-1,1]
so the MLP receives a well-conditioned input; the API works with raw poses.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

POSE_SCALE = 3.0   # position normalisation; covers ≈ ±3 units in training data
POSE_DIM   = 12    # 3 position + 9 orientation


class _SinEmb(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1)
        ).float()
        args = x.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _Emb(nn.Module):
    def __init__(self, emb_dim: int = 64) -> None:
        super().__init__()
        self.sin = _SinEmb(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.sin(x))


class PoseNet(nn.Module):
    """
    DSM / score network in 12D pose space.

    forward(pose_noisy, sigma, pose_0, pose_T, t_frac) → pose_clean

    All tensors are in raw world coordinates (not normalised).
    Internally, positions are divided by POSE_SCALE for numerical stability.
    """

    def __init__(self, hidden: int = 256, emb_dim: int = 64) -> None:
        super().__init__()
        self.noise_emb = _Emb(emb_dim)   # embeds log(sigma)
        self.time_emb  = _Emb(emb_dim)   # embeds t * 1000

        # 3 pose vectors (normalised) + noise emb + time emb
        in_dim = POSE_DIM * 3 + emb_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, POSE_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ── normalisation helpers ────────────────────────────────────────────────

    @staticmethod
    def _norm(pose: Tensor) -> Tensor:
        """Scale positions by 1/POSE_SCALE; leave rotation components unchanged."""
        p = pose.clone()
        p[..., :3] = p[..., :3] / POSE_SCALE
        return p

    @staticmethod
    def _denorm(pose: Tensor) -> Tensor:
        p = pose.clone()
        p[..., :3] = p[..., :3] * POSE_SCALE
        return p

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        pose_noisy: Tensor,   # [B, 12]  raw
        sigma:      Tensor,   # [B]
        pose_0:     Tensor,   # [B, 12]  raw
        pose_T:     Tensor,   # [B, 12]  raw
        t_frac:     Tensor,   # [B]  in [0, 1]
    ) -> Tensor:              # [B, 12]  raw
        noise_e = self.noise_emb(torch.log(sigma.clamp(min=1e-8)))
        time_e  = self.time_emb(t_frac.float() * 1000.0)
        emb     = torch.cat([noise_e, time_e], dim=-1)   # [B, emb_dim*2]

        pn = self._norm(pose_noisy)
        p0 = self._norm(pose_0)
        pT = self._norm(pose_T)

        x = torch.cat([pn, p0, pT, emb], dim=-1)        # [B, 36 + emb_dim*2]
        out_norm = self.net(x)                            # [B, 12] normalised
        return self._denorm(out_norm)                     # raw
