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
        )

    def forward(self, x):
        return self.block(x)


class Autoencoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 64),  # 256 -> 128
            EncoderBlock(64, 128),          # 128 -> 64
            EncoderBlock(128, 256),         # 64 -> 32
            #EncoderBlock(256, 512),         # 32 -> 16
        )

        self.decoder = nn.Sequential(
            #DecoderBlock(512, 256),         # 16 -> 32
            DecoderBlock(256, 128),          # 32 -> 64
            DecoderBlock(128, 64),           # 64 -> 128
            # Final output reconstruction: no BN, Sigmoid activation
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
