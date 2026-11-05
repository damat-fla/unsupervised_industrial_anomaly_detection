from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.v2.functional import gaussian_blur
import torch.nn.functional as F
import torch.nn as nn
import torch

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

    def forward(self, x):
        with torch.no_grad():
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

    def __init__(self, backbone_name: str = "resnet18", feature_layers: list[str] = ["layer2"]):
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
        '''
        super().__init__()
        self.teacher = FeatureExtractor(model_name=backbone_name, layers=feature_layers)
        self.student = FeatureAutoencoder(in_channels=self.teacher.total_channels)
        ...

    def forward(self, x):
        '''
        At train time returns the features extracted by the teacher and the reconstructed features from the student.
        At inference time returns the anomaly map computed by cosine similarity between the teacher and student features.
        '''

        target_feats = self.teacher(x)
        # TODO: handle multiple feature layers by concatenating them along the channel dimension
        target_feat = target_feats['layer2']  # [B, C, H, W] 
        recon_feat = self.student(target_feat) # [B, C, H, W]

        print(f"Target feature shape: {target_feat.shape}, Reconstructed feature shape: {recon_feat.shape}")

        if self.training:
            return target_feat, recon_feat

        return compute_anomaly_map(target_feat, recon_feat)

def compute_anomaly_map(
    target_feat: torch.Tensor, 
    recon_feat: torch.Tensor, 
    out_size: tuple[int, int] = (256, 256),
    blur_sigma: float = 4.0
) -> torch.Tensor:
    """
    Computes the anomaly map based on the cosine similarity between the target features and the reconstructed features.
    """
    # Copute cosine similarity between the target and reconstructed features along the channel dimension
    cos_sim = F.cosine_similarity(target_feat, recon_feat, dim=1) # [B, H_feat, W_feat]

    # Compute the anomaly map as 1 - cosine similarity, and add a channel dimension
    anomaly_map = (1.0 - cos_sim).unsqueeze(1)

    # Resize the anomaly map to the original image size
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode="bilinear", align_corners=False)

    # gaussian blur the anomaly map to smooth it out and reduce noise
    # if blur_sigma is not None and blur_sigma > 0:
    #     kernel_size = int(2 * round(2 * blur_sigma) + 1)
    #     anomaly_map = gaussian_blur(
    #         anomaly_map, 
    #         kernel_size=[kernel_size, kernel_size], 
    #         sigma=[blur_sigma, blur_sigma]
    #    )
    return anomaly_map


