"""
Spatiotemporal 3D UNet for denoising score matching on (H × W × T) volumes.

The full sequence of T-2 intermediate frames is treated as a single 3D volume
[B, 1, Tm, H, W].  The model is conditioned on:
  - noise level σ  (AdaGN in every ResBlock)
  - ctx_first broadcast across the Tm dimension
  - ctx_last  broadcast across the Tm dimension

Architecture
------------
  Input: cat([vol_noisy, ctx_first_broadcast, ctx_last_broadcast], dim=1) → [B, 3, Tm, H, W]
  Encoder:
    init_conv:    Conv3d(3 → ch,    3, p=1)           (Tm, 64, 64)
    enc1:         ResBlock3D(ch  → ch  )               skip s1
    down1:        Conv3d(ch  → 2ch, (1,3,3), stride=(1,2,2))  (Tm, 32, 32)
    enc2:         ResBlock3D(2ch → 2ch )               skip s2
    down2:        Conv3d(2ch → 4ch, (1,3,3), stride=(1,2,2))  (Tm, 16, 16)
    enc3:         ResBlock3D(4ch → 4ch )               skip s3
    down3:        Conv3d(4ch → 4ch, (2,3,3), stride=(2,2,2))  (Tm/2, 8, 8)
    bottleneck:   ResBlock3D(4ch → 4ch )
  Decoder:
    up3:  ConvTranspose3d(4ch, 4ch, (2,2,2), stride=2)  (Tm, 16, 16); cat s3 → ResBlock3D(8ch→4ch)
    up2:  ConvTranspose3d(4ch, 2ch, (1,2,2), stride=(1,2,2))  (Tm, 32, 32); cat s2 → ResBlock3D(4ch→2ch)
    up1:  ConvTranspose3d(2ch, ch,  (1,2,2), stride=(1,2,2))  (Tm, 64, 64); cat s1 → ResBlock3D(2ch→ch)
    out:  GroupNorm → SiLU → Conv3d(ch, 1, 3, p=1)

base_ch=16 keeps peak memory ≈ 30 MB for batch=4, T=16 on CPU.
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


class _NoiseEmb(nn.Module):
    def __init__(self, sin_dim: int = 128, mlp_dim: int = 128) -> None:
        super().__init__()
        self.sin = _SinEmb(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, mlp_dim * 2), nn.SiLU(),
            nn.Linear(mlp_dim * 2, mlp_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        return self.mlp(self.sin(torch.log(sigma.clamp(min=1e-8))))


class ResBlock3D(nn.Module):
    """3D ResBlock with AdaGN noise conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128) -> None:
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        ss = self.emb_proj(F.silu(emb))[:, :, None, None, None]   # [B, 2*ch, 1, 1, 1]
        scale, shift = ss.chunk(2, dim=1)
        h = F.silu(h * (1.0 + scale) + shift)
        h = self.conv2(h)
        return h + self.skip(x)


class STUNet(nn.Module):
    """
    Spatiotemporal UNet: denoises a [B, 1, Tm, H, W] volume of intermediate frames.

    forward(vol_noisy, sigma, ctx_first, ctx_last) → [B, 1, Tm, H, W]
      vol_noisy, ctx_first, ctx_last : all [B, 1, Tm, H, W]
      sigma                          : [B]
    """

    def __init__(self, base_ch: int = 16, emb_dim: int = 128) -> None:
        super().__init__()
        ch = base_ch

        self.noise_emb = _NoiseEmb(sin_dim=128, mlp_dim=emb_dim)

        # Encoder
        self.init_conv = nn.Conv3d(3, ch, 3, padding=1)
        self.enc1  = ResBlock3D(ch,    ch,    emb_dim)
        self.down1 = nn.Conv3d(ch,  ch*2, (1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.enc2  = ResBlock3D(ch*2,  ch*2,  emb_dim)
        self.down2 = nn.Conv3d(ch*2, ch*4, (1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.enc3  = ResBlock3D(ch*4,  ch*4,  emb_dim)
        self.down3 = nn.Conv3d(ch*4, ch*4, (2,3,3), stride=(2,2,2), padding=(0,1,1))

        # Bottleneck
        self.bottleneck = ResBlock3D(ch*4, ch*4, emb_dim)

        # Decoder
        self.up3  = nn.ConvTranspose3d(ch*4, ch*4, (2,2,2), stride=(2,2,2))
        self.dec3 = ResBlock3D(ch*4 + ch*4, ch*4, emb_dim)
        self.up2  = nn.ConvTranspose3d(ch*4, ch*2, (1,2,2), stride=(1,2,2))
        self.dec2 = ResBlock3D(ch*2 + ch*2, ch*2, emb_dim)
        self.up1  = nn.ConvTranspose3d(ch*2, ch,  (1,2,2), stride=(1,2,2))
        self.dec1 = ResBlock3D(ch   + ch,   ch,   emb_dim)

        # Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv3d(ch, 1, 3, padding=1)

    def forward(
        self,
        vol_noisy:  Tensor,   # [B, 1, Tm, H, W]
        sigma:      Tensor,   # [B]
        ctx_first:  Tensor,   # [B, 1, Tm, H, W]
        ctx_last:   Tensor,   # [B, 1, Tm, H, W]
    ) -> Tensor:              # [B, 1, Tm, H, W]
        emb = self.noise_emb(sigma)                                   # [B, emb_dim]

        x = torch.cat([vol_noisy, ctx_first, ctx_last], dim=1)        # [B, 3, Tm, H, W]
        x = self.init_conv(x)                                          # [B, ch, Tm, H, W]

        s1 = self.enc1(x, emb)                                         # [B, ch,   Tm,   64, 64]
        s2 = self.enc2(self.down1(s1), emb)                            # [B, ch*2, Tm,   32, 32]
        s3 = self.enc3(self.down2(s2), emb)                            # [B, ch*4, Tm,   16, 16]
        x  = self.bottleneck(self.down3(s3), emb)                      # [B, ch*4, Tm/2,  8,  8]

        x = self.dec3(torch.cat([self.up3(x), s3], dim=1), emb)       # [B, ch*4, Tm,   16, 16]
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1), emb)       # [B, ch*2, Tm,   32, 32]
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1), emb)       # [B, ch,   Tm,   64, 64]

        return self.out_conv(F.silu(self.out_norm(x)))                 # [B, 1, Tm, H, W]
