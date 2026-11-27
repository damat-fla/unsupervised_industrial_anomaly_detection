from utils.helpers import smooth_anomaly_map
from losses import compute_ssim_map
import torch.nn as nn
import torch


class ConvEncoderBlock(nn.Module):
    """Two 3x3 convs, the first with stride 2: halves the resolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)



class ConvDecoderBlock(nn.Module):
    """Bilinear upsampling x2 followed by a 3x3 conv: doubles the resolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResEncoderBlock(nn.Module):
    """Residual version of ConvEncoderBlock: 1x1 stride-2 conv as shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class ResDecoderBlock(nn.Module):
    """Residual decoder block: upsampling x2, then two 3x3 convs with a 1x1 conv shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        return self.act(self.conv(x) + self.shortcut(x))

class LinearBottleneck(nn.Module):
    """Compresses the bottleneck feature map into a vector of latent_dim values and expands it back."""

    def __init__(self, channels, latent_dim, spatial_size=16):
        super().__init__()
        self.c, self.h, self.w = channels, spatial_size, spatial_size
        self.encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * spatial_size * spatial_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_dim, channels * spatial_size * spatial_size),
            nn.BatchNorm1d(channels * spatial_size * spatial_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Unflatten(1, (channels, spatial_size, spatial_size)),
        )


    def forward(self, x):
        return self.decode(self.encode(x))


class Autoencoder(nn.Module):
    """
    Convolutional autoencoder for pixel reconstruction: 4 encoder blocks (256 -> 16) and a symmetric decoder.
    - block_type: 'conv' or 'residual' blocks.
    - latent_dim: if given, adds a LinearBottleneck of that size.
    """
    def __init__(self, in_channels=3, base_channels=32, latent_dim=None, block_type="conv", img_size=256):
        super().__init__()

        c1 = base_channels

        if block_type == "conv":
            EncoderBlock = ConvEncoderBlock
            DecoderBlock = ConvDecoderBlock
        elif block_type == "residual":
            EncoderBlock = ResEncoderBlock
            DecoderBlock = ResDecoderBlock
        else:
            raise ValueError(f"Unknown block_type '{block_type}'. Choose from 'conv' or 'residual'.")

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, c1),       # 256 -> 128
            EncoderBlock(c1, c1 * 2),          # 128 -> 64
            EncoderBlock(c1 * 2, c1 * 4),      # 64 -> 32
            EncoderBlock(c1 * 4, c1 * 8),      # 32 -> 16
        )

        self.decoder = nn.Sequential(
            DecoderBlock(c1 * 8, c1 * 4),      # 16 -> 32
            DecoderBlock(c1 * 4, c1 * 2),      # 32 -> 64
            DecoderBlock(c1 * 2, c1),          # 64 -> 128
            # Final output reconstruction: 2 convs at 256x256
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        if latent_dim:
            self.bottleneck = LinearBottleneck(
                channels=c1 * 8, 
                latent_dim=latent_dim,
                spatial_size=img_size // 16  # 4 stride-2 encoder blocks
            )
        else:
            self.bottleneck = nn.Identity()

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        x_recon = self.decoder(z)
        return x_recon, x


    def compute_anomaly_map(self, reconstructions, images, method="l1_ssim", alpha=0.5, blur_sigma=4.0):
        """
        Computes anomaly map combining pixel L1 difference and (1 - SSIM) from torchmetrics,
        with torchvision Gaussian smoothing to suppress pixel-level edge noise.
        """
        if method == "l1_ssim":
            l1_map = torch.mean(torch.abs(images - reconstructions), dim=1, keepdim=True)
            ssim_map = torch.mean(compute_ssim_map(images, reconstructions), dim=1, keepdim=True)
            anomaly_map = (1.0 - alpha) * l1_map + alpha * (1.0 - ssim_map)
        else:
            anomaly_map = torch.mean((images - reconstructions) ** 2, dim=1, keepdim=True)

        return smooth_anomaly_map(anomaly_map, blur_sigma)