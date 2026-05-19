"""
model.py
--------
Hybrid CNN-Transformer reconstruction network.

The architecture is a 4-level U-Net with channel widths
[base, 2*base, 4*base, 8*base, 16*base] (default base=16), augmented
by a Transformer self-attention block at the bottleneck. The
transformer treats the (H/16, W/16) bottleneck feature map as a
sequence of 16x16 = 256 tokens of dimension 16*base, lets every
token attend to every other, and folds the result back into a 2D
tensor. The skip connections preserve the U-Net inductive bias for
local CNN refinement at full resolution, while the bottleneck
attention provides the long-range coupling required to inpaint
missing Fourier modes coherently across the whole field.

This is a deliberate scaled-down sketch of the Visual / Metadata-
Aware Transformer style used in the ALMA BRAIN line of work. With
``base=16`` the model has ~4.5 M parameters and fits comfortably on
a single 8 GB GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _ngroups(c: int) -> int:
    """Pick the largest GroupNorm group count <=8 that divides ``c``."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_ngroups(out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_ngroups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    """Bilinear upsample + 3x3 conv + concat skip + ConvBlock.

    Bilinear+conv ("resize-conv", Odena et al. 2016) is used in place
    of ``ConvTranspose2d(kernel=2, stride=2)`` to avoid checkerboard
    artefacts. With small base widths or short training schedules,
    transpose-conv upsampling leaves a very visible grid pattern in
    the output that masquerades as a learned feature; resize-conv
    eliminates it at essentially no cost.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear",
                        align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class TransformerBottleneck(nn.Module):
    """Single Transformer encoder block over flattened spatial tokens."""

    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 4.0, max_tokens: int = 1024):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)               # (B, HW, C)
        seq = seq + self.pos[:, :seq.shape[1]]
        a, _ = self.attn(self.norm1(seq), self.norm1(seq), self.norm1(seq),
                         need_weights=False)
        seq = seq + a
        seq = seq + self.mlp(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class UNetAttn(nn.Module):
    """U-Net with a Transformer self-attention bottleneck.

    Parameters
    ----------
    in_ch
        Input channels (1 for dirty image alone, 2 if dirty + beam
        are concatenated, etc.).
    base
        Channel multiplier; total parameters scale as ``base^2``.
    num_heads
        Attention heads in the bottleneck transformer.
    residual
        If True, learn a correction to the input dirty image rather
        than the full output. This typically converges faster.
    """

    def __init__(self, in_ch: int = 1, base: int = 16,
                 num_heads: int = 4, residual: bool = True):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.inc = ConvBlock(in_ch, c1)
        self.d1 = Down(c1, c2)
        self.d2 = Down(c2, c3)
        self.d3 = Down(c3, c4)
        self.d4 = Down(c4, c5)
        self.bottleneck_attn = TransformerBottleneck(c5, num_heads=num_heads)
        self.u4 = Up(c5, c4, c4)
        self.u3 = Up(c4, c3, c3)
        self.u2 = Up(c3, c2, c2)
        self.u1 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, 1, 1)
        self.residual = residual
        if residual:
            # start as exact identity: out = 0 + dirty. The network has
            # to actively earn any deviation from passing the dirty image
            # through, which makes early training much more stable.
            nn.init.zeros_(self.outc.weight)
            nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.d1(s1)
        s3 = self.d2(s2)
        s4 = self.d3(s3)
        b  = self.d4(s4)
        b  = self.bottleneck_attn(b)
        x4 = self.u4(b, s4)
        x3 = self.u3(x4, s3)
        x2 = self.u2(x3, s2)
        x1 = self.u1(x2, s1)
        out = self.outc(x1)
        if self.residual:
            # the network predicts a correction to the dirty input
            out = out + x[:, :1]
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = UNetAttn(in_ch=1, base=16)
    x = torch.randn(2, 1, 256, 256)
    y = m(x)
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"parameters:   {count_parameters(m):,}")
