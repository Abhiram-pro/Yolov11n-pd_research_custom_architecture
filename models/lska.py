"""
lska.py (updated)

Large Separable Kernel Attention (LSKA) - tolerant split sizes

Changes vs previous:
- If `num_splits` does not evenly divide `channels`, we now either:
  1) automatically choose a sensible divisor from a preferred set [4,3,2,1] if user passed None,
  OR
  2) if user provided num_splits that doesn't divide channels, distribute the remainder
     so that splits sizes are either `base` or `base+1` (first `remainder` splits get +1).
- This prevents AssertionError for common channel sizes (e.g., 256, 512, 768, 1024).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitDWConv(nn.Module):
    """Helper block: depthwise conv with optional 1xK/Kx1 decomposition to save cost.
       Works on a specific channel group (group channels).
    """
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
    Large Separable Kernel Attention (LSKA) - robust split sizes.

    Args:
        channels: input channels
        num_splits: desired number of channel splits/branches (optional). If None,
                    a sensible divisor will be chosen automatically (4, 3, 2, or 1).
                    If provided and does not divide channels evenly, channels will
                    be distributed across splits so each split has either base or base+1 channels.
        large_kernel: the large kernel to use for two decomposed convs (default 21)
        mid_kernel: medium kernel for the third branch (default 7)
        reduction: unused here but kept for API compatibility
    """
    def __init__(self,
                 channels: int,
                 num_splits: Optional[int] = None,
                 large_kernel: int = 21,
                 mid_kernel: int = 7,
                 reduction: int = 16):
        super().__init__()

        # If user didn't specify splits, pick a sensible divisor (prefer 4, then 3, then 2)
        if num_splits is None:
            for d in (4, 3, 2, 1):
                if channels % d == 0:
                    chosen_splits = d
                    break
            else:
                chosen_splits = 1
            num_splits = chosen_splits

        # Compute split sizes that sum to channels.
        # If channels % num_splits != 0, distribute remainder across the first `r` groups.
        base = channels // num_splits
        rem = channels % num_splits
        split_sizes = [base + (1 if i < rem else 0) for i in range(num_splits)]
        assert sum(split_sizes) == channels

        self.channels = channels
        self.num_splits = num_splits
        self.split_sizes = split_sizes

        # Choose kernels for splits. Use large_kernel for first two splits (if exist), mid_kernel for third.
        kernels = []
        for i in range(num_splits):
            if i < 2:
                kernels.append(large_kernel)
            else:
                kernels.append(mid_kernel)

        # Per-split depthwise decomposed convs
        self.split_convs = nn.ModuleList([
            SplitDWConv(ch_size, k, decomposed=True) for ch_size, k in zip(self.split_sizes, kernels)
        ])

        # Fuse conv after concatenation
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(self.split_sizes), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        # Spatial attention: use avg+max pool -> conv -> sigmoid
        self.att_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W] reweighted
        """
        # split according to computed split sizes
        parts = torch.split(x, self.split_sizes, dim=1)
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
