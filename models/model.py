"""
model.py

Top-level assembly for YOLO11n-PD.

Assembles:
- Stem and hierarchical backbone using C3k2Ghost blocks.
- SPPF-like pooling at deepest stage.
- BOTBlock at deepest stage and LSKA attention modules.
- Neck (simple top-down PAN-style fusion) using C3k2Ghost blocks.
- LSCDHead as the detection head (shared conv head).

Factory: build_yolo11n_pd(num_classes=4) -> returns nn.Module ready for forward passes.

Notes:
- This is intended for experimentation and integration into a training loop.
- Keep channels consistent: P3=256, P4=512, P5=1024 (head expects [256,512,1024] by default).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt relative imports first; fallback to direct imports if run from models dir
try:
    from .ghost import GhostConv, GhostBottleneck, count_params
    from .c3k2_ghost import C3k2Ghost, ConvBNAct
    from .bot import BOTBlock
    from .lska import LSKA
    from .lscd_head import LSCDHead
except Exception:
    from ghost import GhostConv, GhostBottleneck, count_params
    from c3k2_ghost import C3k2Ghost, ConvBNAct
    from bot import BOTBlock
    from lska import LSKA
    from lscd_head import LSCDHead


# ---------------------------
# Small helpers for downsample blocks
# ---------------------------
class DownSampleBlock(nn.Module):
    """
    Downsample block: Conv (stride=2) then C3k2Ghost block.
    Produces feature maps at progressively smaller spatial sizes.
    """
    def __init__(self, in_c: int, out_c: int, c3_layers: int = 1):
        super().__init__()
        self.down = ConvBNAct(in_c, out_c, k=3, s=2)
        self.c3 = C3k2Ghost(out_c, out_c, n=c3_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.c3(x)
        return x


# ---------------------------
# YOLO11n-PD assembly
# ---------------------------
class YOLO11nPD(nn.Module):
    def __init__(self, num_classes: int = 4):
        """
        Build the YOLO11n-PD model.

        Args:
            num_classes: number of classes for detection (e.g., 4 for RDD2022).
        """
        super().__init__()
        # Stem
        self.stem = ConvBNAct(3, 64, k=3, s=2)   # 640 -> 320

        # Backbone (progressive downsampling)
        # Keep C3k2Ghost blocks shallow to remain lightweight
        self.stage1 = C3k2Ghost(64, 64, n=1)     # 320
        self.down1 = DownSampleBlock(64, 128, c3_layers=1)   # 160
        self.down2 = DownSampleBlock(128, 256, c3_layers=1)  # 80  -> P3
        self.down3 = DownSampleBlock(256, 512, c3_layers=1)  # 40  -> P4
        self.down4 = DownSampleBlock(512, 512, c3_layers=1)  # 20  -> pre-P5

        # lightweight SPPF-like pooling (stacked maxpool)
        self.sppf = nn.Sequential(
            nn.MaxPool2d(5, stride=1, padding=2),
            nn.MaxPool2d(5, stride=1, padding=2)
        )

        # PD modules at deepest level
        self.bot = BOTBlock(512, hidden_dim=512, num_heads=4, expansion=1.0)
        self.lska_deep = LSKA(512, num_splits=3, large_kernel=21)

        # Neck (top-down + bottom-up simplified PAN)
        # Use C3k2Ghost to keep efficient
        self.p5_fuse = C3k2Ghost(512, 512, n=1)   # refine P5 features
        self.p4_fuse = C3k2Ghost(1024, 512, n=1)  # after concat p5_up + p4
        self.p3_fuse = C3k2Ghost(768, 256, n=1)   # after concat p4_up + p3 -> reduce to 256

        # LSKA on fused maps
        self.lska_mid = LSKA(512, num_splits=3, large_kernel=21)
        self.lska_shallow = LSKA(256, num_splits=3, large_kernel=21)

        # expand p5 channels to 1024 for head (mimic concatenation growth)
        self.expand_p5 = ConvBNAct(512, 1024, k=1)

        # Head expects [P3(256), P4(512), P5(1024)]
        self.head = LSCDHead([256, 512, 1024], num_classes=num_classes, shared_c=256, use_aux_conv=False)

    def forward(self, x: torch.Tensor):
        # Stem + shallow stage
        x = self.stem(x)           # 320x320x64
        x = self.stage1(x)         # still 320

        # Progressive downsampling
        p2 = self.down1(x)         # 160x160x128
        p3 = self.down2(p2)        # 80x80x256  -> P3
        p4 = self.down3(p3)        # 40x40x512  -> P4
        p5 = self.down4(p4)        # 20x20x512  -> pre-P5

        # SPPF-like: increase receptive field without huge cost
        p5 = p5 + self.sppf(p5)

        # BOT + LSKA at deepest level
        p5 = self.bot(p5)          # global context
        p5 = self.lska_deep(p5)    # large-kernel attention

        # Top-down path: P5 -> up -> concat P4 -> fuse -> P4_out
        p5_refined = self.p5_fuse(p5)                # refine p5
        p5_up = F.interpolate(p5_refined, scale_factor=2, mode='nearest')  # 40x40
        # ensure shapes compatible for concat; if channel counts differ, we still concat and fuse
        f4 = torch.cat([p5_up, p4], dim=1)           # [B, 512+512=1024, 40,40]
        f4 = self.lska_mid(f4) if f4.shape[1] == 1024 else f4
        p4_out = self.p4_fuse(f4)                    # make into 512 ch

        # Next top-down: p4_out -> up -> concat p3 -> fuse -> p3_out
        p4_up = F.interpolate(p4_out, scale_factor=2, mode='nearest')      # 80x80
        f3 = torch.cat([p4_up, p3], dim=1)           # [B, 512+256=768, 80,80]
        f3 = self.lska_shallow(f3) if f3.shape[1] == 768 else f3
        p3_out = self.p3_fuse(f3)                    # reduce to 256 ch

        # Expand p5 for head usage
        p5_exp = self.expand_p5(p5_refined)          # make p5 -> 1024 ch

        # Final feature maps to head: p3_out (256), p4_out (512), p5_exp (1024)
        feats = [p3_out, p4_out, p5_exp]
        preds = self.head(feats)
        return preds


# Factory
def build_yolo11n_pd(num_classes: int = 4) -> nn.Module:
    """
    Factory to create a YOLO11n-PD model instance.
    """
    model = YOLO11nPD(num_classes=num_classes)
    return model


# Quick smoke test when run as a script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = build_yolo11n_pd(num_classes=4).to(device)
    model.eval()
    # dummy input
    x = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        preds = model(x)
    print("Number of output scales:", len(preds))
    for i, p in enumerate(preds):
        print(f"Scale {i}: shape {p.shape}")
    print("Total params:", sum(p.numel() for p in model.parameters()))
