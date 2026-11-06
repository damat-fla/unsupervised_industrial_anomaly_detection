from torchvision.transforms.v2.functional import gaussian_blur
from losses import compute_ssim_map
import torch.nn as nn
import torch


class EncoderBlock(nn.Module):
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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with an optional 1x1 conv shortcut when channels or spatial resolution change.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.shortcut(x) + self.conv(x))


class ResEncoderBlock(nn.Module):
    """
    Encoder block: downsamples by 2 using a stride-2 residual block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        return self.block(x)


class ResDecoderBlock(nn.Module):
    """
    Decoder block: upsamples by 2 then refines with a residual block.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res_block = ResidualBlock(in_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.res_block(x)


class BottleneckSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention block for 2D feature maps at bottleneck resolution.
    Maintains spatial resolution (sharpness) while allowing every patch to attend
    to all other patches and absolute positions (learning relational template rules).
    """
    def __init__(self, channels, num_heads=4, feat_size=(16, 16)):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels * 2, channels),
        )   

        # Learnable 2D positional embeddings so patches know absolute spatial location
        h, w = feat_size
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten spatial dimensions into a sequence of tokens: (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        # Multi-Head Self-Attention with residual connection
        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out

        # Feed-Forward Network with residual connection
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Reshape back to 2D feature map: (B, C, H, W)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out


class ResAutoencoder(nn.Module):
    """
    Flexible Residual Autoencoder with configurable bottleneck:
      - bottleneck_type='none': pure convolutional bottleneck (sharp, no extra constraints)
      - bottleneck_type='linear': 1D dense vector bottleneck (strong global template, but can be blurrier)
      - bottleneck_type='attention': 2D Multi-Head Self-Attention + Positional Encoding (template rules + sharp)
    """
    def __init__(self, in_channels=3, base_channels=32, bottleneck_type="attention", latent_dim=128, num_attn_heads=4):
        super().__init__()

        c1 = base_channels
        self.bottleneck_channels = c1 * 8
        self.bottleneck_type = bottleneck_type.lower()

        self.encoder = nn.Sequential(
            ResEncoderBlock(in_channels, c1),      # 256 -> 128
            ResEncoderBlock(c1, c1 * 2),          # 128 -> 64
            ResEncoderBlock(c1 * 2, c1 * 4),      # 64 -> 32
            ResEncoderBlock(c1 * 4, self.bottleneck_channels),  # 32 -> 16
        )

        if self.bottleneck_type == "attention":
            self.bottleneck = BottleneckSelfAttention(
                channels=self.bottleneck_channels,
                num_heads=num_attn_heads,
                feat_size=(16, 16),
            )
        elif self.bottleneck_type == "linear":
            self.to_latent = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(self.bottleneck_channels * 4 * 4, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.from_latent = nn.Sequential(
                nn.Linear(latent_dim, self.bottleneck_channels * 4 * 4),
                nn.BatchNorm1d(self.bottleneck_channels * 4 * 4),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Unflatten(1, (self.bottleneck_channels, 4, 4)),
                nn.Upsample(size=(16, 16), mode="bilinear", align_corners=False),
            )
        elif self.bottleneck_type == "none":
            self.bottleneck = nn.Identity()
        else:
            raise ValueError(f"Unknown bottleneck_type '{bottleneck_type}'. Choose from 'none', 'linear', or 'attention'.")

        self.decoder = nn.Sequential(
            ResDecoderBlock(self.bottleneck_channels, c1 * 4),  # 16 -> 32
            ResDecoderBlock(c1 * 4, c1 * 2),                   # 32 -> 64
            ResDecoderBlock(c1 * 2, c1),                       # 64 -> 128
            # Final output reconstruction: upsample to 256x256 + convs
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.encoder(x)
        if self.bottleneck_type == "linear":
            z = self.to_latent(feat)
            feat = self.from_latent(z)
        else:
            feat = self.bottleneck(feat)
        return self.decoder(feat)


class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        c1 = base_channels

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, c1),       # 256 -> 128
            EncoderBlock(c1, c1 * 2),          # 128 -> 64
            EncoderBlock(c1 * 2, c1 * 4),      # 64 -> 32
            EncoderBlock(c1 * 4, c1 * 8),      # 32 -> 16
            # AdvancedEncoderBlock(in_channels, c1),       # 256 -> 128
            # AdvancedEncoderBlock(c1, c1 * 2),          # 128 -> 64
            # AdvancedEncoderBlock(c1 * 2, c1 * 4),      # 64 -> 32
            # AdvancedEncoderBlock(c1 * 4, c1 * 8)    # 32 -> 16
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

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, x

    def compute_anomaly_map(self, x, x_recon):
        """
        Computes anomaly map combining pixel L1 difference and (1 - SSIM) from torchmetrics,
        with torchvision Gaussian smoothing to suppress pixel-level edge noise.
        """
        anomaly_map = ...

        return anomaly_map

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

        if blur_sigma is not None and blur_sigma > 0:
            kernel_size = int(2 * round(2 * blur_sigma) + 1)
            anomaly_map = gaussian_blur(anomaly_map, kernel_size=[kernel_size, kernel_size], sigma=[blur_sigma, blur_sigma])

        return anomaly_map