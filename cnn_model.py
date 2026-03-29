import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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


class ResConvBlock(nn.Module):
    """Conv block with residual skip; 1x1 projection when channels differ."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.skip = (nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        out = self.relu(out + self.skip(x))
        return self.drop(out)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ResConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ResConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # pad if spatial sizes don't match (odd input dims)
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet3D(nn.Module):
    """3D U-Net: (B, C, 64, 64, 64) -> (B, 1, 64, 64, 64) log10(nH2)."""

    def __init__(self, n_channels: int = 15, base_ch: int = 16, dropout: float = 0.1):
        super().__init__()
        b = base_ch
        self.enc1 = ResConvBlock(n_channels, b,   dropout=dropout)
        self.enc2 = Down(b,   b*2,               dropout=dropout)
        self.enc3 = Down(b*2, b*4,               dropout=dropout)
        # bottleneck gets 2x dropout
        self.bot  = Down(b*4, b*8,               dropout=min(dropout*2, 0.5))
        self.dec3 = Up(b*8, b*4, b*4,            dropout=dropout)
        self.dec2 = Up(b*4, b*2, b*2,            dropout=dropout)
        self.dec1 = Up(b*2, b,   b,              dropout=dropout)
        self.out  = nn.Conv3d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.bot(s3)
        x  = self.dec3(x, s3)
        x  = self.dec2(x, s2)
        x  = self.dec1(x, s1)
        return self.out(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet3D(n_channels=15, base_ch=16)
    print(f"UNet3D parameters: {count_parameters(model):,}")

    x = torch.randn(1, 15, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    assert y.shape == (1, 1, 64, 64, 64), f"Unexpected output shape: {y.shape}"
    print("OK")
