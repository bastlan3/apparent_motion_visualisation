import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, log_sigma: Tensor) -> Tensor:
        # log_sigma: [B] → [B, dim]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=log_sigma.device) / max(half - 1, 1)
        )
        args = log_sigma[:, None] * freqs[None, :]   # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


class NoiseEmbedding(nn.Module):
    def __init__(self, sin_dim: int = 128, mlp_dim: int = 128) -> None:
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, mlp_dim * 2),
            nn.SiLU(),
            nn.Linear(mlp_dim * 2, mlp_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        # sigma: [B] strictly positive → [B, mlp_dim]
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        return self.mlp(self.sin_emb(log_sigma))


class TimeEmbedding(nn.Module):
    def __init__(self, sin_dim: int = 128, mlp_dim: int = 128) -> None:
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, mlp_dim * 2),
            nn.SiLU(),
            nn.Linear(mlp_dim * 2, mlp_dim),
        )

    def forward(self, t_frac: Tensor) -> Tensor:
        # t_frac: [B] in [0, 1] → [B, mlp_dim]
        # Scale to [0, 1000] so sinusoidal frequencies cover the same dynamic range
        # as DDPM-style integer timesteps
        return self.mlp(self.sin_emb(t_frac * 1000.0))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        # emb_proj: Linear → scale + shift for AdaGN; zero-init for stable start
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        h = self.norm2(h)
        # AdaGN: scale+shift from noise embedding
        ss = self.emb_proj(F.silu(emb))[:, :, None, None]   # [B, out_ch*2, 1, 1]
        scale, shift = ss.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class CondUNet(nn.Module):
    """
    Conditional UNet for denoising score matching.

    Inputs:
        x_noisy:   [B, 1, H, W]  noisy/corrupted intermediate frame
        sigma:     [B]           noise level (strictly positive)
        ctx_first: [B, 1, H, W]  first frame (always clean)
        ctx_last:  [B, 1, H, W]  last frame (always clean)
        t_frac:    [B]           normalised time position in [0, 1]
                                 (t_frac = t / (T-1); 0=first, 1=last)
    Output:
        [B, 1, H, W]  predicted clean frame (Tweedie MAP estimate)
    """

    def __init__(self, base_ch: int = 32, emb_dim: int = 128) -> None:
        super().__init__()
        ch = base_ch

        self.noise_emb = NoiseEmbedding(sin_dim=128, mlp_dim=emb_dim)
        self.time_emb  = TimeEmbedding(sin_dim=128,  mlp_dim=emb_dim)

        # Encoder
        self.init_conv = nn.Conv2d(3, ch, 3, padding=1)          # 3 channels: noisy+ctx1+ctx2
        self.enc1 = ResBlock(ch,    ch,    emb_dim)               # 64×64, ch
        self.down1 = nn.Conv2d(ch,  ch*2, 3, stride=2, padding=1) # 32×32, ch*2
        self.enc2 = ResBlock(ch*2,  ch*2,  emb_dim)               # 32×32, ch*2
        self.down2 = nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1) # 16×16, ch*4
        self.enc3 = ResBlock(ch*4,  ch*4,  emb_dim)               # 16×16, ch*4
        self.down3 = nn.Conv2d(ch*4, ch*4, 3, stride=2, padding=1) # 8×8, ch*4

        # Bottleneck
        self.bottleneck = ResBlock(ch*4, ch*4, emb_dim)

        # Decoder (skip = concat after upsample)
        self.up3  = nn.ConvTranspose2d(ch*4, ch*4, 2, stride=2)   # 8→16, ch*4
        self.dec3 = ResBlock(ch*4 + ch*4, ch*4, emb_dim)           # 256→128
        self.up2  = nn.ConvTranspose2d(ch*4, ch*2, 2, stride=2)   # 16→32, ch*2
        self.dec2 = ResBlock(ch*2 + ch*2, ch*2, emb_dim)           # 128→64
        self.up1  = nn.ConvTranspose2d(ch*2, ch,  2, stride=2)    # 32→64, ch
        self.dec1 = ResBlock(ch   + ch,   ch,   emb_dim)           # 64→32

        # Output head
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(
        self,
        x_noisy:   Tensor,
        sigma:     Tensor,
        ctx_first: Tensor,
        ctx_last:  Tensor,
        t_frac:    Tensor | None = None,
    ) -> Tensor:
        emb = self.noise_emb(sigma)                                   # [B, emb_dim]
        if t_frac is not None:
            emb = emb + self.time_emb(t_frac)                        # additive conditioning

        x = torch.cat([x_noisy, ctx_first, ctx_last], dim=1)         # [B, 3, H, W]
        x = self.init_conv(x)                                         # [B, ch, H, W]

        # Encoder + skip connections
        s1 = self.enc1(x, emb)                                        # [B, ch, 64, 64]
        s2 = self.enc2(self.down1(s1), emb)                           # [B, ch*2, 32, 32]
        s3 = self.enc3(self.down2(s2), emb)                           # [B, ch*4, 16, 16]
        x  = self.bottleneck(self.down3(s3), emb)                     # [B, ch*4, 8, 8]

        # Decoder
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1), emb)      # [B, ch*4, 16, 16]
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1), emb)      # [B, ch*2, 32, 32]
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1), emb)      # [B, ch, 64, 64]

        return self.out_conv(F.silu(self.out_norm(x)))                # [B, 1, H, W]
