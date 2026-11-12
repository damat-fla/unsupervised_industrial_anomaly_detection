from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms.v2.functional import gaussian_blur, normalize
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
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Choose one of available options: 'resnet18', 'resnet50'.")

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
                 per_layer_cosine: bool = False, fusion: str = "concat", fpn_channels: int | None = None):
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
            - fusion: str,
                how the selected layers are fused into the student input. The reconstruction target is
                always the (fixed) concatenation of the upsampled teacher features.
                * "concat": the student input is the target itself.
                * "fpn": the student input is the finest output of a trainable Feature Pyramid Network
                  (torchvision implementation of Lin et al., 2017) applied to the teacher features.
            - fpn_channels: int,
                output channels of the FPN (default: total number of teacher channels). If smaller than the
                total, the FPN compresses the features and a 1x1 conv projects them back to the total number
                of channels, so that the student always has the same size as with "concat".
        '''
        super().__init__()
        if fusion not in ("concat", "fpn"):
            raise ValueError(f"Unknown fusion '{fusion}'. Choose from 'concat' or 'fpn'.")
        # the FPN fuses levels top-down, so layers must go from the finest to the deepest
        if fusion == "fpn" and feature_layers != sorted(feature_layers):
            raise ValueError(f"With fusion='fpn' feature_layers must be ordered from shallow to deep, got {feature_layers}.")

        self.teacher = FeatureExtractor(model_name=backbone_name, layers=feature_layers)
        total_channels = self.teacher.total_channels
        self.fusion = fusion
        if fusion == "fpn":
            fpn_channels = fpn_channels or total_channels
            self.fpn = FeaturePyramidNetwork(in_channels_list=list(self.teacher.out_channels.values()),
                                             out_channels=fpn_channels)
            # 1x1 projection back to the student input size when the FPN compresses the channels
            self.fpn_proj = (nn.Conv2d(fpn_channels, total_channels, kernel_size=1)
                             if fpn_channels != total_channels else nn.Identity())
        self.student = FeatureAutoencoder(in_channels=total_channels)
        self.img_size = img_size  # anomaly maps are upsampled to the input image size
        # Channels of each selected layer, in concatenation order (None = single global cosine)
        self.channels_per_layer = list(self.teacher.out_channels.values()) if per_layer_cosine else None

    def forward(self, x):
        '''
        At train time returns the features extracted by the teacher and the reconstructed features from the student.
        At inference time returns the anomaly map computed by cosine similarity between the teacher and student features.
        '''

        teacher_feats = self.teacher(x)  # dict: layer name -> [B, C_l, H_l, W_l]
        feats = list(teacher_feats.values())
        size = max(f.shape[-2:] for f in feats)  # highest resolution among the selected layers
        # Target: every layer upsampled to the same resolution and concatenated along the channel dimension
        target_feat = torch.cat(
            [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats], dim=1
        )  # [B, total_channels, H, W]

        if self.fusion == "fpn":
            # finest FPN level (P2 for layer2+layer3) [B, fpn_channels, H, W], projected to [B, total_channels, H, W]
            student_input = self.fpn_proj(next(iter(self.fpn(teacher_feats).values())))
        else:
            student_input = target_feat
        recon_feat = self.student(student_input)  # [B, total_channels, H, W]

        return recon_feat, target_feat

    def compute_anomaly_map(
        self,
        recon_feat: torch.Tensor,
        target_feat: torch.Tensor,
        blur_sigma: float = 4.0
    ) -> torch.Tensor:
        """
        Computes the anomaly map based on the cosine similarity between the target features and the reconstructed features.
        """
        # 1 - cosine similarity along the channel dimension (global or per layer, same as the loss),
        # plus a channel dimension: [B, 1, H_feat, W_feat]
        anomaly_map = cosine_distance_map(recon_feat, target_feat, self.channels_per_layer).unsqueeze(1)

        # Resize the anomaly map to the original image size
        anomaly_map = F.interpolate(anomaly_map, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # gaussian blur the anomaly map to smooth it out and reduce noise
        # if blur_sigma is not None and blur_sigma > 0:
        #     kernel_size = int(2 * round(2 * blur_sigma) + 1)
        #     anomaly_map = gaussian_blur(
        #         anomaly_map, 
        #         kernel_size=[kernel_size, kernel_size], 
        #         sigma=[blur_sigma, blur_sigma]
        #    )
        return anomaly_map


