"""
c3k2_ghost.py

C3k2Ghost: a C3-like module (C3k2) implemented using GhostBottleneck primitives.

Design notes:
- Mirrors the C3/C3k2 pattern used in YOLO-series backbones: split input into two branches,
  process one branch through a stack of bottlenecks, then concat and fuse.
- Replaces regular bottlenecks with GhostBottleneck to reduce params/FLOPs.
- Exposes a simple interface: C3k2Ghost(in_channels, out_channels, n=1, expansion=0.5)
- Uses ConvBNAct (simple Conv->BN->SiLU) for the entry/exit projections.

Intended usage:
- Replace standard C3/C3k2 blocks in backbone and neck with this efficient variant.
- Import this module in top-level model builder.
"""

from typing import List
import torch
import torch.nn as nn

# Import Ghost primitives and a ConvBNAct helper if present.
# Expect models/ghost.py to be sibling module under models/
try:
    from models.ghost import GhostBottleneck, GhostConv, count_params
except Exception:
    # fallback relative import if running as script from models folder
    from ghost import GhostBottleneck, GhostConv, count_params

# Local helper: Conv -> BN -> SiLU
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2Ghost(nn.Module):
    """
    C3k2Ghost module.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        n (int): number of GhostBottleneck bottlenecks in the middle branch
        expansion (float): hidden expansion ratio relative to out_channels (default 0.5)
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, expansion: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.expansion = expansion
        hidden_channels = int(round(out_channels * expansion))

        # Entry conv branches
        self.cv1 = ConvBNAct(in_channels, hidden_channels, k=1)
        self.cv2 = ConvBNAct(in_channels, hidden_channels, k=1)

        # stack of GhostBottleneck modules
        bottlenecks: List[nn.Module] = []
        for _ in range(n):
            # We'll expand -> project inside GhostBottleneck such that hidden==hidden
            bottlenecks.append(GhostBottleneck(hidden_channels, hidden_channels, hidden_channels, stride=1))
        self.m = nn.Sequential(*bottlenecks)

        # fusion conv
        self.conv_fuse = ConvBNAct(2 * hidden_channels, out_channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # branch 1: conv -> stack -> out1
        y1 = self.cv1(x)
        y1 = self.m(y1)
        # branch 2: direct conv (identity path) -> out2
        y2 = self.cv2(x)
        # concat and fuse
        y = torch.cat((y1, y2), dim=1)
        return self.conv_fuse(y)


# quick smoke test when run directly
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    m = C3k2Ghost(64, 128, n=2, expansion=0.5).to(device)
    x = torch.randn(2, 64, 80, 80).to(device)
    y = m(x)
    print("C3k2Ghost out shape:", y.shape)  # expect [2,128,80,80]
    try:
        print("Params:", count_params(m))
    except Exception:
        print("Params unavailable (count_params import failed).")
