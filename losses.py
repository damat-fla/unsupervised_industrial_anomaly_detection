import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure


def compute_ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise SSIM map between x and y (values in [0, 1]) using torchmetrics.
    Returns a tensor of shape (B, C, H, W).
    """
    _, ssim_map = structural_similarity_index_measure(x, y, data_range=1.0, return_full_image=True)
    return ssim_map


def cosine_distance_map(pred: torch.Tensor, target: torch.Tensor, channels_per_layer: list[int] | None = None) -> torch.Tensor:
    """
    Compute 1 - cosine similarity along the channel dimension. Returns a tensor of shape (B, H, W).
    - channels_per_layer: if None, a single cosine is computed on all channels (layers concatenated).
      Otherwise the channels are split per layer (e.g. [128, 256]), one cosine is computed for each
      layer and the per-layer distances are averaged, so that every layer has the same weight.
    """
    if channels_per_layer is None:
        return 1.0 - F.cosine_similarity(pred, target, dim=1)
    layer_pairs = zip(pred.split(channels_per_layer, dim=1), target.split(channels_per_layer, dim=1))
    distances = [1.0 - F.cosine_similarity(p, t, dim=1) for p, t in layer_pairs]
    return torch.stack(distances).mean(dim=0)


class L1SSIMLoss(nn.Module):
    """
    Combined L1 + SSIM loss widely used for MVTec-AD anomaly detection:
        loss = (1 - alpha) * L1 + alpha * (1 - SSIM)
    Default alpha = 0.5 balances absolute pixel differences and structural similarity.
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
        return (1.0 - self.alpha) * l1 + self.alpha * (1.0 - ssim_val)

class CosineLoss(nn.Module):
    """
    Cosine similarity loss for feature reconstruction.
    Computes 1 - cosine_similarity(pred, target) averaged over batch and spatial dimensions.
    - channels_per_layer: if given, the cosine is computed per layer and averaged (see cosine_distance_map).
    """
    def __init__(self, channels_per_layer: list[int] | None = None):
        super().__init__()
        self.channels_per_layer = channels_per_layer

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cosine_distance_map(pred, target, self.channels_per_layer).mean()


class MultiLayerCosineLoss(nn.Module):
    """
    Cosine loss for lists of per-layer feature maps, possibly at different resolutions
    (MultiHeadFeatureReconstructionModel): 1 - cosine similarity averaged over the positions of each layer,
    then averaged over the layers, so that every layer has the same weight.
    """
    def forward(self, pred: list[torch.Tensor], target: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([cosine_distance_map(p, t).mean() for p, t in zip(pred, target)]).mean()

class MSELoss(nn.Module):
    """
    Mean Squared Error loss for feature reconstruction.
    Computes the average squared difference between pred and target.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

