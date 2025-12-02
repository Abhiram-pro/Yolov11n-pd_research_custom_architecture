"""
Fully dynamic LSKA module.
- Supports any channel count at runtime.
- Automatically rebuilds split sizes and depthwise conv branches.
- Prevents split_with_sizes errors (e.g., 512 vs 1024 mismatches).
- Supports decomposed large kernel convs (k×1 + 1×k) for efficiency.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# SplitDWConv  (depthwise conv per channel-split)
# ---------------------------------------------------------

class SplitDWConv(nn.Module):
    def __init__(self, channels: int, kernel: int, decomposed: bool = True):
        super().__init__()
        assert isinstance(channels, int) and channels > 0
        assert isinstance(kernel, int) and kernel >= 1

        pad = kernel // 2

        if decomposed and kernel > 1:
            # Use (k×1) then (1×k) to approximate large-kernel depthwise conv
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, (kernel, 1), padding=(pad, 0),
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),

                nn.Conv2d(channels, channels, (1, kernel), padding=(0, pad),
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
        else:
            # Direct depthwise conv
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel, padding=pad,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# LSKA  (Large Separable Kernel Attention)
# ---------------------------------------------------------

class LSKA(nn.Module):
    def __init__(
        self,
        channels: int,
        num_splits: Optional[int] = None,
        large_kernel: int = 21,
        mid_kernel: int = 7,
        decomposed: bool = True,
        reduction: int = 16
    ):
        super().__init__()

        # ------------------------
        # Store config
        # ------------------------
        self.channels = channels
        self.num_splits_user = num_splits
        self.large_kernel = large_kernel
        self.mid_kernel = mid_kernel
        self.decomposed = decomposed
        self.reduction = reduction

        # Build initial configuration
        self._build_splits_and_layers(channels)

    # -----------------------------------------------------
    # Internal builder: creates split sizes, per-branch convs,
    # and fuse conv based on current channel count
    # -----------------------------------------------------
    def _build_splits_and_layers(self, C: int):

        # 1) Determine number of splits
        if self.num_splits_user is None:
            # auto choose a divisor: prefer 4, then 3, then 2, then 1
            for d in (4, 3, 2, 1):
                if C % d == 0:
                    num_splits = d
                    break
            else:
                num_splits = 1
        else:
            num_splits = self.num_splits_user

        self.num_splits = num_splits

        # 2) Compute split sizes even if not divisible
        base = C // num_splits
        rem = C % num_splits
        self.split_sizes = [base + (1 if i < rem else 0) for i in range(num_splits)]

        # 3) Determine kernel per split (first 2: large, rest: mid)
        kernels = []
        for i in range(num_splits):
            kernels.append(self.large_kernel if i < 2 else self.mid_kernel)

        # 4) Build depthwise conv branches
        self.split_convs = nn.ModuleList([
            SplitDWConv(split_c, k, decomposed=self.decomposed)
            for split_c, k in zip(self.split_sizes, kernels)
        ])

        # 5) Build fuse (1×1 conv)
        self.fuse = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
        )

        # 6) Attention conv (avg+max descriptor)
        self.att_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Keep track of configured channels
        self.channels = C

    # -----------------------------------------------------
    # Runtime channel-adaptive rebuild
    # -----------------------------------------------------
    def _rebuild_for_channels(self, new_C: int):
        """Rebuilds split sizes and conv layers dynamically to match new channel count."""
        # Get the device of the current module
        device = next(self.parameters()).device
        self._build_splits_and_layers(new_C)
        # Move newly created layers to the same device
        self.to(device)

    # -----------------------------------------------------
    # Forward pass
    # -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        output: [B, C, H, W]
        """

        B, C, H, W = x.shape

        # Auto-rebuild if incoming channels do NOT match current config
        if C != self.channels:
            self._rebuild_for_channels(C)

        # Split input into feature groups
        parts = torch.split(x, self.split_sizes, dim=1)

        # Apply depthwise convs per split
        outs = []
        for p, conv in zip(parts, self.split_convs):
            outs.append(conv(p))

        # Concat + fuse
        u = torch.cat(outs, dim=1)
        u = self.fuse(u)

        # Spatial attention
        sa_avg = u.mean(dim=1, keepdim=True)
        sa_max, _ = u.max(dim=1, keepdim=True)
        sa = torch.cat([sa_avg, sa_max], dim=1)

        w = self.att_conv(sa)     # shape [B,1,H,W]
        out = x * w               # weighted original features

        return out

    def __repr__(self):
        return (
            f"LSKA(channels={self.channels}, "
            f"num_splits={self.num_splits}, "
            f"split_sizes={self.split_sizes}, "
            f"large_kernel={self.large_kernel}, "
            f"mid_kernel={self.mid_kernel}, "
            f"decomposed={self.decomposed})"
        )
