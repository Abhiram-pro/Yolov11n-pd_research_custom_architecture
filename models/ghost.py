"""
ghost.py
GhostNet primitives used by the YOLO11n-PD model.

Provides:
- GhostConv: lightweight conv that generates "ghost" features cheaply.
- GhostBottleneck: bottleneck using GhostConv (expand -> depthwise -> project).
- small utilities: count_params (for quick sanity checks)

Notes:
- Implemented with BatchNorm + SiLU activations to match the rest of the model.
- Intended to be imported by c3k2_ghost.py and model.py later.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostConv(nn.Module):
    """
    Ghost convolution block.

    Produces primary features via a regular Conv then cheap 'ghost' features via a
    depthwise conv on the primary features. Concatenates primary + ghost to produce
    `out_channels` features.

    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size for primary conv (default 1)
        ratio: ratio between primary channels and total (default 2 -> primary ~ out/2)
        dw_kernel: depthwise kernel size for cheap op (default 3)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 ratio: int = 2,
                 dw_kernel: int = 3,
                 use_bn: bool = True):
        super().__init__()
        assert ratio >= 1, "ratio must be >= 1"
        self.out_channels = out_channels
        # primary channels are ceil(out / ratio)
        self.prim_channels = math.ceil(out_channels / ratio)
        self.ghost_channels = out_channels - self.prim_channels

        # primary conv (pointwise by default)
        pad = (kernel_size - 1) // 2
        self.primary = nn.Conv2d(in_channels, self.prim_channels, kernel_size,
                                 stride=1, padding=pad, bias=False)
        self.primary_bn = nn.BatchNorm2d(self.prim_channels) if use_bn else nn.Identity()

        # cheap operation: depthwise conv on primary features
        pad_dw = (dw_kernel - 1) // 2
        if self.ghost_channels > 0:
            self.cheap = nn.Conv2d(self.prim_channels, self.ghost_channels,
                                   dw_kernel, stride=1, padding=pad_dw,
                                   groups=self.prim_channels, bias=False)
            self.cheap_bn = nn.BatchNorm2d(self.ghost_channels) if use_bn else nn.Identity()
        else:
            self.cheap = None
            self.cheap_bn = None

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # primary features
        p = self.primary(x)
        p = self.primary_bn(p)
        if self.cheap is not None:
            g = self.cheap(p)
            g = self.cheap_bn(g)
            out = torch.cat((p, g), dim=1)
        else:
            out = p
        # trim if extra channels (defensive)
        if out.shape[1] != self.out_channels:
            out = out[:, :self.out_channels, :, :].contiguous()
        return self.act(out)


class GhostBottleneck(nn.Module):
    """
    GhostBottleneck block.

    Structure:
    - GhostConv(in -> hidden)
    - Depthwise conv (stride may be >1)
    - GhostConv(hidden -> out)
    - Residual connection (project if needed)

    Args:
        in_channels: input channels
        hidden_channels: expanded hidden channels (often out_channels or expansion*in)
        out_channels: output channels
        stride: stride for the depthwise conv (1 or 2)
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.stride = stride
        self.ghost1 = GhostConv(in_channels, hidden_channels, kernel_size=1, ratio=2, dw_kernel=3)
        if stride > 1:
            # keep it efficient: depthwise conv to reduce spatial dim, then BN+SiLU
            self.dw = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1,
                                groups=hidden_channels, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_channels)
        else:
            self.dw = None
            self.dw_bn = None
        self.ghost2 = GhostConv(hidden_channels, out_channels, kernel_size=1, ratio=2, dw_kernel=3)

        # shortcut / projection
        if stride != 1 or in_channels != out_channels:
            # Use a lightweight projection for the shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ghost1(x)
        if self.dw is not None:
            y = self.dw_bn(self.dw(y))
        y = self.ghost2(y)
        return y + self.shortcut(x)


# small utility
def count_params(m: nn.Module) -> int:
    """Return the number of trainable parameters in module m."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# quick internal smoke test when running file directly
if __name__ == "__main__":
    # quick forward/backward smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Test GhostConv
    gc = GhostConv(64, 128, kernel_size=1, ratio=2, dw_kernel=3).to(device)
    x = torch.randn(2, 64, 80, 80).to(device)
    y = gc(x)
    print("GhostConv out shape:", y.shape)  # expected [2,128,80,80]
    print("GhostConv params:", count_params(gc))

    # Test GhostBottleneck
    gb = GhostBottleneck(128, 128, 128, stride=1).to(device)
    x2 = torch.randn(2, 128, 40, 40).to(device)
    y2 = gb(x2)
    print("GhostBottleneck out shape:", y2.shape)  # expected [2,128,40,40]
    print("GhostBottleneck params:", count_params(gb))
