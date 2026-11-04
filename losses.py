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
