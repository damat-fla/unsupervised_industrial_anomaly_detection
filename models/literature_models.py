"""
Wrappers around the torch models of anomalib for the methods from the literature (PatchCore, PaDiM).

These methods do not train any weight: they fit statistics of the features of a frozen backbone
(a memory bank of patch features for PatchCore) with a single pass over the training images.
The wrappers only adapt anomalib to the protocol of the project:
  - inputs are images in [0, 1] (as produced by MVTec2DDataset): ImageNet normalization is applied here,
    as in the FeatureExtractor of the custom model;
  - fit(loader, device) builds the memory bank;
  - predict(images) returns the anomaly map [B, 1, H, W] and the native image score [B] of the method
    (anomaly map smoothing and score are the original ones of anomalib).
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as v2
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from models.feature_autoencoder import IMAGENET_MEAN, IMAGENET_STD


class PatchCoreWrapper(nn.Module):
    def __init__(self, backbone: str = "resnet18.tv_in1k", layers: list[str] = ["layer2", "layer3"],
                 coreset_sampling_ratio: float = 0.1, num_neighbors: int = 9):
        """
        Parameters:
        -----------
            - backbone: str, timm name of the backbone. "resnet18.tv_in1k" = torchvision IMAGENET1K_V1 weights,
              the same as the teacher of the custom model (plain "resnet18" in timm loads different weights).
            - layers: list[str], backbone layers whose features are concatenated into the patch embedding.
            - coreset_sampling_ratio: float, fraction of the training patches kept in the memory bank.
            - num_neighbors: int, neighbors used to re-weight the image score (as in the paper).
        """
        super().__init__()
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.model = PatchcoreModel(layers=list(layers), backbone=backbone, pre_trained=True,
                                    num_neighbors=num_neighbors)

    @torch.no_grad()
    def fit(self, loader: torch.utils.data.DataLoader, device: torch.device):
        """
        Single pass over the training images: stores the patch embeddings, then keeps a coreset of them as the
        memory bank. The loader must not drop any image (drop_last=False).
        """
        # In anomalib, train() is only a flag telling PatchcoreModel.forward to store the embeddings.
        # The backbone stays frozen: TimmFeatureExtractor forces eval() on it at every forward,
        # so the BatchNorm statistics are never updated.
        self.model.train()
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            self.model(v2.normalize(images, mean=IMAGENET_MEAN, std=IMAGENET_STD))

        self.model.subsample_embedding(self.coreset_sampling_ratio)  # coreset (random start: set the seed before)
        self.model.eval()

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        images: [B, 3, H, W] in [0, 1]. Returns the anomaly map [B, 1, H, W] (nearest neighbor distance of each
        patch, upsampled and smoothed with sigma 4) and the native image score [B] (re-weighted max patch score).
        """
        self.model.eval()
        output = self.model(v2.normalize(images, mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return output.anomaly_map, output.pred_score
