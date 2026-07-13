"""
Simple CNN encoder-decoder for underwater image restoration.

This is the first deep-learning baseline in the restoration progression
(simple CNN -> UWCNN/WaterNet -> GAN as last resort). It uses light skip
connections (U-Net style, 3 levels) to avoid the blurring typical of pure
bottleneck encoder-decoders, while staying small and cheap compared to
domain-specific architectures.

Expects input tensors of shape (batch, 3, H, W), float32, normalized to
[0, 1], with H and W divisible by 8 (three downsampling stages, each
halving spatial size).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU- The basic building block used everywhere
    in this network, except for the down/up sampling steps themselves.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Extracts features at the current resolution (kept as a skip
    connection), then halves spatial size with a stride-2 convolution.
    
    A strided conv is used instead of MaxPool so the downsampling itself
    is learned, rather than a fixed, non-differentiable operation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.downsample(skip)
        return down, skip
    

class UpBlock(nn.Module):
    """Doubles spatial size with a transposed convolution, concatenates the
    result with the matching skip connection from the encoder, then fuses
    both with a ConvBlock.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        # *2 because we concatenate the unsampled tensor with the skip
        # connection, doubling thte channel count before fusing.
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    

class RestorationCNN(nn.Module):
    """Simple 3-level encoder-decoder with skip connections for underwater
    image restoration. First deep-learning baseline in the projec's
    restoration progression.
    """

    def __init__(self):
        super().__init__()

        self.down1 = DownBlock(3, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)

        self.bottleneck = ConvBlock(128, 256)

        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)

        # Final conv to get back to 3 channels (RGB)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)

        bottleneck = self.bottleneck(x3)

        u3 = self.up3(bottleneck, skip3)
        u2 = self.up2(u3, skip2)
        u1 = self.up1(u2, skip1)

        restored = self.output_conv(u1)
        return torch.sigmoid(restored)  # Ensure output is in [0, 1]
    

if __name__ == "__main__":
    model = RestorationCNN()
    dummy_input = torch.rand(2, 3, 256, 256)  # batch of 2, RGB, 256x256
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")