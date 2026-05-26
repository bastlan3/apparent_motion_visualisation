"""
FlowNet: predicts the velocity field v(x_t, t) for conditional flow matching.

The model is conditioned on:
  - temporal position t ∈ [0,1] via TimeEmbedding + AdaGN
  - boundary frames ctx_first and ctx_last concatenated as input channels

At each time t the model outputs dx/dt ≈ v_θ(x_t, t; ctx_first, ctx_last).
Integrating this ODE from t=0 to t=1 reconstructs the full trajectory.

Architecture mirrors CondUNet (same ResBlock / AdaGN pattern) but the
embedding is purely temporal — no noise-level conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _SinEmb(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device) / max(half - 1, 1)
        ).float()
        args = x.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Embeds t ∈ [0,1] into a dense vector of size emb_dim."""
    def __init__(self, emb_dim: int = 128) -> None:
        super().__init__()
        self.sin = _SinEmb(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sin(t * 1000.0))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        ss = self.emb_proj(F.silu(emb))[:, :, None, None]
        scale, shift = ss.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class FlowNet(nn.Module):
    """
    Conditional flow matching network for apparent-motion sequences.

    Inputs
    ------
    x_t       : [B, 1, H, W]  current frame (or zeros when frame is missing)
    t         : [B]           normalised time in [0, 1]
    ctx_first : [B, 1, H, W]  first frame of sequence (always observed)
    ctx_last  : [B, 1, H, W]  last frame of sequence  (always observed)

    Output
    ------
    velocity  : [B, 1, H, W]  dx/dt at the given (x, t)

    The linear interpolant  x_lin(t) = (1-t)*ctx_first + t*ctx_last
    is concatenated to the input so the model can easily represent the
    "straight-line" flow as a simple residual.
    """

    def __init__(self, base_ch: int = 32, emb_dim: int = 128) -> None:
        super().__init__()
        ch = base_ch

        self.time_emb = TimeEmbedding(emb_dim)

        # 4 input channels: x_t, ctx_first, ctx_last, linear_interp
        self.init_conv = nn.Conv2d(4, ch, 3, padding=1)

        self.enc1 = ResBlock(ch,    ch,    emb_dim)
        self.down1 = nn.Conv2d(ch,  ch*2,  3, stride=2, padding=1)
        self.enc2 = ResBlock(ch*2,  ch*2,  emb_dim)
        self.down2 = nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1)
        self.enc3 = ResBlock(ch*4,  ch*4,  emb_dim)
        self.down3 = nn.Conv2d(ch*4, ch*4, 3, stride=2, padding=1)
        self.bottleneck = ResBlock(ch*4, ch*4, emb_dim)

        self.up3  = nn.ConvTranspose2d(ch*4, ch*4, 2, stride=2)
        self.dec3 = ResBlock(ch*8, ch*4, emb_dim)
        self.up2  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)
        self.dec2 = ResBlock(ch*4, ch*2, emb_dim)
        self.up1  = nn.ConvTranspose2d(ch*2, ch,   2, stride=2)
        self.dec1 = ResBlock(ch*2, ch,   emb_dim)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_t:       Tensor,
        t:         Tensor,
        ctx_first: Tensor,
        ctx_last:  Tensor,
    ) -> Tensor:
        t = t.float()
        emb = self.time_emb(t)

        # Straight-line interpolant as an extra hint channel
        x_lin = (1.0 - t[:, None, None, None]) * ctx_first + t[:, None, None, None] * ctx_last

        x = torch.cat([x_t, ctx_first, ctx_last, x_lin], dim=1)
        x = self.init_conv(x)

        s1 = self.enc1(x, emb)
        s2 = self.enc2(self.down1(s1), emb)
        s3 = self.enc3(self.down2(s2), emb)
        x  = self.bottleneck(self.down3(s3), emb)

        x = self.dec3(torch.cat([self.up3(x), s3], dim=1), emb)
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1), emb)
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(x)))
