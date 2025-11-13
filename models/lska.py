"""
lska.py

Large Separable Kernel Attention (LSKA)

LSKA provides large spatial receptive field attention at relatively low cost by:
- Splitting channels into groups (num_splits).
- Applying different large-kernel depthwise operations per split (e.g., 1xK, Kx1 decompositions).
- Concatenating outputs, fusing, and computing spatial attention using pooled descriptors (avg+max).
- Reweighting original input features by the generated spatial attention map.

Design goals:
- Capture long, thin structures (cracks) and extended context.
- Keep FLOPs manageable by using depthwise separable / decomposed large kernels.
- Be plug-and-play at backbone tail and neck fusion points.

Usage:
    from models.lska import LSKA
    m = LSKA(channels=512, num_splits=3, large_kernel=21)
    out = m(x)  # x: (B, 512, H, W)

Tweakable parameters:
- num_splits: number of channel sub-groups to apply different kernel sizes to.
- large_kernel: main large kernel size used for two decomposed convs (k x 1 and 1 x k).
- mid_kernel: smaller kernel used for an additional branch (defaults to 7).
- reduction: final channel reduction factor for the attention conv.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitDWConv(nn.Module):
    """Helper block: depthwise conv with optional 1xK/Kx1 decomposition to save cost."""
    def __init__(self, channels: int, kernel: int, decomposed: bool = True):
        super().__init__()
        pad = kernel // 2
        if decomposed and kernel > 1:
            # implement as (k x 1) followed by (1 x k) depthwise conv to reduce cost
            self.conv1 = nn.Conv2d(channels, channels, (kernel, 1), padding=(pad, 0), groups=channels, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, (1, kernel), padding=(0, pad), groups=channels, bias=False)
            self.net = nn.Sequential(
                self.conv1,
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                self.conv2,
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            # direct depthwise conv
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel, padding=pad, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class LSKA(nn.Module):
    """
    Large Separable Kernel Attention (LSKA)

    Args:
        channels: input channels
        num_splits: number of channel splits/branches (default 3)
        large_kernel: the large kernel to use for splits (default 21)
        mid_kernel: medium kernel for the third branch (default 7)
        reduction: intermediate channels reduction for attention conv (default 16)
    """
    def __init__(self,
                 channels: int,
                 num_splits: int = 3,
                 large_kernel: int = 21,
                 mid_kernel: int = 7,
                 reduction: int = 16):
        super().__init__()
        assert channels % num_splits == 0, "channels must be divisible by num_splits"
        self.channels = channels
        self.num_splits = num_splits
        self.split_c = channels // num_splits

        # Choose kernels for splits
        kernels = []
        if num_splits >= 1:
            kernels.append(large_kernel)
        if num_splits >= 2:
            kernels.append(large_kernel)
        if num_splits >= 3:
            kernels.append(mid_kernel)
        # If more splits requested, reuse mid_kernel for extras
        while len(kernels) < num_splits:
            kernels.append(mid_kernel)

        # Per-split depthwise decomposed convs
        self.split_convs = nn.ModuleList([SplitDWConv(self.split_c, k, decomposed=True) for k in kernels])

        # Fuse conv after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(self.split_c * num_splits, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        # Spatial attention: use avg+max pool -> conv -> sigmoid
        self.att_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # optional channel reduction for computational control (not used to alter shape)
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W] reweighted
        """
        parts = torch.split(x, self.split_c, dim=1)
        outs: List[torch.Tensor] = []
        # apply per-split large depthwise convs
        for p, conv in zip(parts, self.split_convs):
            outs.append(conv(p))
        # concat and fuse
        u = torch.cat(outs, dim=1)           # [B, C, H, W]
        u = self.fuse(u)                     # [B, C, H, W]

        # spatial descriptors
        sa_avg = torch.mean(u, dim=1, keepdim=True)   # [B,1,H,W]
        sa_max, _ = torch.max(u, dim=1, keepdim=True) # [B,1,H,W]
        sa = torch.cat([sa_avg, sa_max], dim=1)       # [B,2,H,W]

        w = self.att_conv(sa)                         # [B,1,H,W] in (0,1)
        out = x * w                                   # reweight original features
        return out


# quick smoke test if run as script
if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    m = LSKA(channels=256, num_splits=3, large_kernel=21, mid_kernel=7).to(device)
    x = torch.randn(2, 256, 80, 80).to(device)
    y = m(x)
    print("LSKA out:", y.shape)  # expect [2,256,80,80]
