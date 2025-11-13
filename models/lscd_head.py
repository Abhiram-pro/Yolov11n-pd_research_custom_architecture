"""
lscd_head.py

LSCD-Head (Lightweight Shared Convolution Detection Head)

Purpose:
- Replace per-scale heavy head stacks with a shared convolutional block.
- Provide per-scale adaptors (ScaleLayer) so the shared conv output can be tuned
  for each scale before final small heads (cls/reg/obj).
- Output format: list of 3 tensors (for P3, P4, P5) each shaped (B, 4+1+nc, H, W).

Design notes:
- Uses GroupNorm in the shared block for stability w.r.t small batch sizes.
- Projects incoming feature maps to a shared channel dimension when needed.
- Minimal final heads (1x1 convs) to keep compute small.
- Includes optional small auxiliary convs per-scale if more capacity is needed.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleLayer(nn.Module):
    """Per-scale learnable scalar multiplier."""
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LSCDHead(nn.Module):
    """
    Lightweight shared detection head.

    Args:
        in_channels_list: list of ints for incoming channels at [P3, P4, P5]
        num_classes: number of object classes (nc)
        shared_c: internal shared channel width (if None, uses min(in_channels_list))
        use_aux_conv: whether to include a small extra conv after scale for per-scale specialization
    """
    def __init__(self,
                 in_channels_list: List[int],
                 num_classes: int,
                 shared_c: Optional[int] = None,
                 use_aux_conv: bool = False):
        super().__init__()
        assert len(in_channels_list) == 3, "Expect three scales (P3, P4, P5)"
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        # choose shared channel width
        self.shared_c = shared_c or min(in_channels_list)
        # per-scale projection to shared channels (1x1 convs or identity)
        self.proj_convs = nn.ModuleList()
        for c in in_channels_list:
            if c != self.shared_c:
                self.proj_convs.append(nn.Conv2d(c, self.shared_c, kernel_size=1, bias=False))
            else:
                self.proj_convs.append(nn.Identity())

        # shared conv block: GroupNorm -> Conv3x3 -> SiLU
        # choose groups for GN: 32 typical, but cannot exceed channels; fallback to 16 or 8
        gn_groups = 32 if self.shared_c >= 32 else (16 if self.shared_c >= 16 else 8)
        self.shared = nn.Sequential(
            nn.GroupNorm(gn_groups, self.shared_c),
            nn.Conv2d(self.shared_c, self.shared_c, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True)
        )

        # per-scale scale layers and optional aux convs
        self.scales = nn.ModuleList([ScaleLayer(1.0) for _ in in_channels_list])
        self.aux_convs = nn.ModuleList()
        for _ in in_channels_list:
            if use_aux_conv:
                self.aux_convs.append(nn.Sequential(
                    nn.Conv2d(self.shared_c, self.shared_c, 3, 1, 1, bias=False),
                    nn.GroupNorm(gn_groups, self.shared_c),
                    nn.SiLU(inplace=True)
                ))
            else:
                self.aux_convs.append(nn.Identity())

        # final small heads per-scale: reg (4), obj (1), cls (nc)
        self.reg_heads = nn.ModuleList([nn.Conv2d(self.shared_c, 4, kernel_size=1) for _ in in_channels_list])
        self.obj_heads = nn.ModuleList([nn.Conv2d(self.shared_c, 1, kernel_size=1) for _ in in_channels_list])
        self.cls_heads = nn.ModuleList([nn.Conv2d(self.shared_c, num_classes, kernel_size=1) for _ in in_channels_list])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        feats: list of 3 tensors [P3 (H3xW3), P4, P5] in increasing stride order
        returns: list of 3 tensors each (B, 4+1+nc, H, W)
        """
        assert len(feats) == 3, "Expect three feature maps"
        outs = []
        for i, x in enumerate(feats):
            # project to shared channels if needed
            x = self.proj_convs[i](x)
            # shared conv
            y = self.shared(x)
            # per-scale scaling & optional aux conv
            y = self.scales[i](y)
            y = self.aux_convs[i](y)
            # final heads
            reg = self.reg_heads[i](y)
            obj = self.obj_heads[i](y)
            cls = self.cls_heads[i](y)
            out = torch.cat([reg, obj, cls], dim=1)
            outs.append(out)
        return outs


# smoke test when run as script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Example incoming channels: [256, 512, 1024] typical for P3, P4, P5
    head = LSCDHead([256, 512, 1024], num_classes=4, shared_c=256, use_aux_conv=True).to(device)
    # generate dummy feature maps with matching channels (or different channels; prog_convs handle it)
    p3 = torch.randn(1, 256, 80, 80).to(device)
    p4 = torch.randn(1, 512, 40, 40).to(device)
    p5 = torch.randn(1, 1024, 20, 20).to(device)
    outs = head([p3, p4, p5])
    print("Outputs lengths:", len(outs))
    for o in outs:
        print("Out shape:", o.shape)  # expect (1, 4+1+nc, H, W)
