from torchvision.models import (resnet18, resnet50, wide_resnet50_2,
                                ResNet18_Weights, ResNet50_Weights, Wide_ResNet50_2_Weights)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.v2.functional import normalize
from utils.helpers import smooth_anomaly_map
from losses import cosine_distance_map
import torch.nn.functional as F
import torch.nn as nn
import torch

# Statistics used to normalize the inputs of torchvision ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.1, inplace=True),
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

# Teacher: a wrapper around a pretrained feature extractor (e.g., ResNet-18)
class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "resnet18", layers: list[str] = ["layer2"]):
        super().__init__()

        self.model_name = model_name.lower()
        self.layers = layers

        if model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "wide_resnet50_2":
            # V1 weights: the ones used by PatchCore/PaDiM, for a like-for-like comparison
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Choose one of available options: "
                             f"'resnet18', 'resnet50', 'wide_resnet50_2'.")

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers)

        # dummy forward pass to determine the total number of output channels for the selected layers
        with torch.no_grad():
            # Small, single-channel input to make a fast forward pass
            dummy_input = torch.empty(1, 3, 64, 64)
            dummy_out = self.feature_extractor(dummy_input)
            
            self.out_channels = {k: v.shape[1] for k, v in dummy_out.items()}
            self.total_channels = sum(self.out_channels.values())

        self.eval()

    def train(self, mode: bool = True):
        # The teacher is frozen: always keep it in eval mode, even when the parent model
        # calls model.train(). Otherwise BatchNorm layers would normalize with batch
        # statistics and update their running_mean/var during training.
        # requires_grad=False only stops gradient updates, not BatchNorm buffer updates.
        return super().train(False)

    def forward(self, x):
        # x: images in [0, 1], shape [B, 3, H, W].
        # ImageNet normalization is applied here (not in the dataset transforms), so the
        # dataset stays in [0, 1], shared with the pixel autoencoder and the visualizations.
        with torch.no_grad():
            x = normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            features = self.feature_extractor(x)
        return features


# Student: a feature autoencoder that learns to reconstruct the features extracted by the teacher
class FeatureAutoencoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, in_channels*2),
            EncoderBlock(in_channels*2, in_channels*4),
        )

        self.decoder = nn.Sequential(
            DecoderBlock(in_channels*4, in_channels*2)
        )

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


# Feature Reconstruction Model: combines the teacher and student for end-to-end training
class FeatureReconstructionModel(nn.Module):

    def __init__(self, backbone_name: str = "resnet18", feature_layers: list[str] = ["layer2"], img_size: int = 256,
                 per_layer_cosine: bool = False, blur_sigma: float = 0.0):
        '''
            A teacher-student model for feature reconstruction. The teacher is a pretrained feature extractor 
            (e.g., ResNet-18), and the student is a feature autoencoder that learns to reconstruct the features 
            extracted by the teacher.
        
            Parameters
            ---------
            - backbone_name: str, 
                name of the pretrained model to use as the teacher.
            - feature_layers: 
                list of str, names of the layers from which to extract features.
            - img_size: int,
                size of the input images (assumed to be square) for upsampling the anomaly maps.
            - per_layer_cosine: bool,
                if True, the cosine distance (loss and anomaly map) is computed per layer and averaged,
                so that every layer has the same weight. If False, a single cosine is computed on the
                concatenated features. Identical when a single layer is used.
            - blur_sigma: float,
                sigma of the Gaussian smoothing of the anomaly map (default 0: no smoothing).
                Used only by compute_anomaly_map: it does not affect training or the loss.
        '''
        super().__init__()
        self.teacher = FeatureExtractor(model_name=backbone_name, layers=feature_layers)
        self.student = FeatureAutoencoder(in_channels=self.teacher.total_channels)
        self.img_size = img_size  # anomaly maps are upsampled to the input image size
        # Channels of each selected layer, in concatenation order (None = single global cosine)
        self.channels_per_layer = list(self.teacher.out_channels.values()) if per_layer_cosine else None
        self.blur_sigma = blur_sigma

    def forward(self, x):
        '''
        At train time returns the features extracted by the teacher and the reconstructed features from the student.
        At inference time returns the anomaly map computed by cosine similarity between the teacher and student features.
        '''

        feats = list(self.teacher(x).values())
        size = max(f.shape[-2:] for f in feats)  # highest resolution among the selected layers
        # Upsample every layer to the same resolution and concatenate along the channel dimension
        target_feat = torch.cat(
            [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats], dim=1
        )  # [B, total_channels, H, W]
        recon_feat = self.student(target_feat)  # [B, total_channels, H, W]

        return recon_feat, target_feat

    def compute_anomaly_map(self, recon_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """
        Computes the anomaly map based on the cosine similarity between the target features and the reconstructed
        features, upsampled to the image size and smoothed with self.blur_sigma. Returns [B, 1, img_size, img_size].
        """
        # 1 - cosine similarity along the channel dimension (global or per layer, same as the loss),
        # plus a channel dimension: [B, 1, H_feat, W_feat]
        anomaly_map = cosine_distance_map(recon_feat, target_feat, self.channels_per_layer).unsqueeze(1)

        # Resize the anomaly map to the original image size
        anomaly_map = F.interpolate(anomaly_map, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return smooth_anomaly_map(anomaly_map, self.blur_sigma)


# Decoder head of the multi-head student: reconstructs one teacher layer at its native resolution
class DecoderHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # same structure as the single-head decoder: 2x upsampling block, then a conv to the layer channels
        self.block = DecoderBlock(in_channels, out_channels * 2)
        self.out_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, z, size):
        x = self.block(z)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # to the native resolution of the layer
        return self.out_conv(x)


# Multi-head student: shared encoder on the concatenated features + one decoder head per teacher layer
class MultiHeadFeatureAutoencoder(nn.Module):
    def __init__(self, in_channels: int, channels_per_layer: list[int], bottleneck_channels: int | None = None,
                 bottleneck_dropout: float = 0.0):
        super().__init__()
        bottleneck_channels = bottleneck_channels or in_channels * 4
        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, in_channels * 2),
            EncoderBlock(in_channels * 2, bottleneck_channels),
        )
        # dropout on the bottleneck z (active only in train mode); Identity when disabled
        self.dropout = nn.Dropout(bottleneck_dropout) if bottleneck_dropout > 0 else nn.Identity()
        self.heads = nn.ModuleList(DecoderHead(bottleneck_channels, c) for c in channels_per_layer)

    def forward(self, x, sizes):
        z = self.dropout(self.encoder(x))
        return [head(z, size) for head, size in zip(self.heads, sizes)]


# Multi-head Feature Reconstruction Model (variant inspired by RD4AD, Deng & Li, CVPR 2022)
class MultiHeadFeatureReconstructionModel(nn.Module):

    def __init__(self, backbone_name: str = "resnet18", feature_layers: list[str] = ["layer2", "layer3"],
                 img_size: int = 256, bottleneck_channels: int | None = None, bottleneck_dropout: float = 0.0,
                 blur_sigma: float = 0.0):
        '''
            Teacher-student feature reconstruction with one decoder head per teacher layer.
            The student input is the concatenation of the teacher layers upsampled to the highest resolution
            (as in FeatureReconstructionModel); a shared encoder compresses it and each head reconstructs one
            layer at its native resolution. The targets are the raw teacher features of each layer.

            Parameters
            ---------
            - backbone_name: str,
                name of the pretrained model to use as the teacher.
            - feature_layers:
                list of str, names of the layers from which to extract features.
            - img_size: int,
                size of the input images (assumed to be square) for upsampling the anomaly maps.
            - bottleneck_channels: int,
                channels of the bottleneck z at the end of the shared encoder (default: 4x the input channels).
                A smaller value compresses z, limiting the capacity of the student to copy anomalies.
            - bottleneck_dropout: float,
                dropout probability on z during training (default 0: disabled). The student has to reconstruct
                from incomplete information, which discourages copying the input (identity shortcut).
            - blur_sigma: float,
                sigma of the Gaussian smoothing of the anomaly map (default 0: no smoothing).
                Used only by compute_anomaly_map: it does not affect training or the loss.
        '''
        super().__init__()
        self.teacher = FeatureExtractor(model_name=backbone_name, layers=feature_layers)
        self.student = MultiHeadFeatureAutoencoder(in_channels=self.teacher.total_channels,
                                                   channels_per_layer=list(self.teacher.out_channels.values()),
                                                   bottleneck_channels=bottleneck_channels,
                                                   bottleneck_dropout=bottleneck_dropout)
        self.img_size = img_size  # anomaly maps are upsampled to the input image size
        self.blur_sigma = blur_sigma

    def forward(self, x):
        '''
        Returns two lists with one tensor per layer, each at the native resolution of the layer:
        the reconstructed features [B, C_l, H_l, W_l] and the teacher features (targets).
        '''
        target_feats = list(self.teacher(x).values())
        size = max(f.shape[-2:] for f in target_feats)  # highest resolution among the selected layers
        student_input = torch.cat(
            [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in target_feats], dim=1
        )  # [B, total_channels, H, W]
        recon_feats = self.student(student_input, [f.shape[-2:] for f in target_feats])
        return recon_feats, target_feats

    def compute_anomaly_map(self, recon_feats: list[torch.Tensor], target_feats: list[torch.Tensor]) -> torch.Tensor:
        """
        One cosine distance map per layer at its native resolution, upsampled to the image size and averaged,
        so that every layer has the same weight, then smoothed with self.blur_sigma. Returns [B, 1, img_size, img_size].
        """
        maps = [
            F.interpolate(cosine_distance_map(r, t).unsqueeze(1), size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
            for r, t in zip(recon_feats, target_feats)
        ]
        return smooth_anomaly_map(torch.stack(maps).mean(dim=0), self.blur_sigma)


