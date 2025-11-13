"""
bot.py

BOTBlock: Lightweight Bottleneck Transformer module suitable for insertion at
the deep backbone output (low spatial resolution, e.g., 20x20). Designed to be
computationally efficient while providing global context for long-range structures.

Contents:
- MHSA: multi-head self-attention operating on flattened spatial tokens.
- FeedForward: small MLP used after attention (adds capacity but keeps it light).
- BOTBlock: full block (proj_in -> LN -> MHSA -> FFN -> proj_out) with residuals.

Design choices:
- Use LayerNorm on token dimension (after flatten) for stable attention.
- Use small expansion factor in feedforward to limit FLOPs.
- Keep number of heads configurable; default 4.
- Keep hidden expansion moderate (expansion=1.0 by default) — you can reduce for lower compute.

Intended placement:
- Insert at the last backbone stage (after SPPF) where HxW is small (e.g., 20x20).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    """
    Multi-Head Self-Attention operating on token sequences [B, N, C].
    - Uses a single linear to produce qkv and a projection out.
    - Implements scaled dot-product attention per head.
    """
    def __init__(self, dim: int, num_heads: int = 4, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Lightweight feed-forward MLP used after attention.
    Implements: Linear -> activation -> Dropout -> Linear
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 1.0)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BOTBlock(nn.Module):
    """
    BOTBlock: bottleneck transformer block intended to be used at the end of the backbone.
    Structure:
        - proj_in: 1x1 conv to reduce/increase channels -> BN -> activation
        - flatten spatial (B, C, H, W) -> (B, N, C)
        - LayerNorm -> MHSA -> add residual (with token projection if needed)
        - LayerNorm -> FeedForward -> add residual
        - unflatten -> proj_out (1x1 conv) -> BN -> residual to original input channels
    """
    def __init__(self,
                 channels: int,
                 hidden_dim: Optional[int] = None,
                 num_heads: int = 4,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
                 expansion: float = 1.0):
        """
        Args:
            channels: input and output channels for the block
            hidden_dim: token embedding dim for MHSA; if None -> channels * expansion
            num_heads: number of attention heads
            expansion: multiplier for the token embedding dim compared to channels
        """
        super().__init__()
        hidden_dim = hidden_dim or int(channels * expansion)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # projections
        self.proj_in = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.proj_in_bn = nn.BatchNorm2d(hidden_dim)
        self.proj_out = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.proj_out_bn = nn.BatchNorm2d(channels)

        # attention on flattened tokens
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MHSA(hidden_dim, num_heads=num_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, hidden_dim, dropout=ffn_dropout)

        # small bottleneck residual projection for token path if channels != hidden_dim
        if hidden_dim != channels:
            self.token_proj = nn.Linear(channels, hidden_dim, bias=False)
            self.token_unproj = nn.Linear(hidden_dim, channels, bias=False)
        else:
            self.token_proj = None
            self.token_unproj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # project in spatial domain
        y = self.proj_in(x)
        y = self.proj_in_bn(y)
        # flatten spatial tokens
        y_flat = y.flatten(2).transpose(1, 2)  # [B, N, hidden_dim]
        # optionally project tokens from C->hidden_dim
        if self.token_proj is not None:
            y_flat = self.token_proj(y_flat)
        # LayerNorm + MHSA
        y_norm = self.norm1(y_flat)
        y_att = self.attn(y_norm)
        y = y_flat + y_att  # residual at token level
        # FFN
        y_norm2 = self.norm2(y)
        y_ffn = self.ffn(y_norm2)
        y = y + y_ffn
        # optionally unproject tokens back to conv channels
        if self.token_unproj is not None:
            y = self.token_unproj(y)
        # reshape back to [B, hidden_dim, H, W]
        y = y.transpose(1, 2).view(B, -1, H, W)
        # project out to original channels
        y = self.proj_out(y)
        y = self.proj_out_bn(y)
        # final residual to input (note: input x channels == output channels)
        out = x + y
        return out


# quick smoke when running as script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # small smoke with 512 channels at 20x20 resolution
    blk = BOTBlock(channels=512, hidden_dim=512, num_heads=4, expansion=1.0).to(device)
    x = torch.randn(1, 512, 20, 20).to(device)
    y = blk(x)
    print("BOTBlock out:", y.shape)  # expect [1,512,20,20]
    # test smaller hidden dim
    blk2 = BOTBlock(channels=256, hidden_dim=256, num_heads=4).to(device)
    x2 = torch.randn(2, 256, 20, 20).to(device)
    y2 = blk2(x2)
    print("BOTBlock out (2):", y2.shape)
