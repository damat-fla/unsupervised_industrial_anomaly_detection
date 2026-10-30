import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        c1 = base_channels

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, c1),       # 256 -> 128
            EncoderBlock(c1, c1 * 2),          # 128 -> 64
            EncoderBlock(c1 * 2, c1 * 4),      # 64 -> 32
            #EncoderBlock(c1 * 4, c1 * 8),      # 32 -> 16
        )

        self.decoder = nn.Sequential(
            #DecoderBlock(c1 * 8, c1 * 4),      # 16 -> 32
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

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon