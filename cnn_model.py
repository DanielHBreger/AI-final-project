"""
cnn_model.py
Small 3D U-Net that maps (B, C, D, H, W) input field volumes to a
(B, 1, D, H, W) predicted log10(fh2) field.

Architecture (encoder-decoder with skip connections):
  Encoder:    conv(C→32) → conv(32→64) → pool
              conv(64→128) → pool
  Bottleneck: conv(128→256)
  Decoder:    up + conv(256+128→128) → conv(128→64)
              up + conv(64+32→32)    → conv(32→1)

Input grid is expected to be 64×64×64 (downsampled from 128×128×128).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two Conv3d layers with InstanceNorm and ReLU, plus optional Dropout3d."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool then ConvBlock."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Trilinear upsample, concatenate skip, then ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # Pad if spatial sizes differ (e.g. odd input dimensions)
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([x, skip], dim=1))


# ── U-Net ─────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    3D U-Net predicting log10(fh2) from multi-channel physical field volumes.

    Args:
        n_channels : number of input feature channels (default 14)
        base_ch    : base number of feature maps (default 32)
        dropout    : Dropout3d rate applied after each ConvBlock (default 0.10);
                     the bottleneck uses 2× this rate for stronger regularisation
    """

    def __init__(self, n_channels: int = 14, base_ch: int = 32, dropout: float = 0.10):
        super().__init__()
        b = base_ch
        # Encoder
        self.enc1 = ConvBlock(n_channels, b,   dropout=dropout)    # (B, b,   D, H, W)
        self.enc2 = Down(b,   b*2,             dropout=dropout)    # (B, b*2, D/2, ...)
        self.enc3 = Down(b*2, b*4,             dropout=dropout)    # (B, b*4, D/4, ...)
        # Bottleneck — stronger dropout to regularise the most compressed representation
        self.bot  = Down(b*4, b*8,             dropout=dropout*2)  # (B, b*8, D/8, ...)
        # Decoder
        self.dec3 = Up(b*8, b*4, b*4,          dropout=dropout)    # (B, b*4, D/4, ...)
        self.dec2 = Up(b*4, b*2, b*2,          dropout=dropout)    # (B, b*2, D/2, ...)
        self.dec1 = Up(b*2, b,   b,            dropout=dropout)    # (B, b,   D,   ...)
        # Output
        self.out  = nn.Conv3d(b, 1, kernel_size=1)                 # (B, 1,   D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.bot(s3)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        return self.out(x)   # raw log10(fh2) prediction


# ── Parameter count helper ─────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = UNet3D(n_channels=14, base_ch=32, dropout=0.10)
    print(f"UNet3D parameters: {count_parameters(model):,}")

    x = torch.randn(1, 14, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    assert y.shape == (1, 1, 64, 64, 64), f"Unexpected output shape: {y.shape}"
    print("Smoke test passed.")
