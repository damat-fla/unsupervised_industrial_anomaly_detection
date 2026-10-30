import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def compute_ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute pixel-wise SSIM map between x and y (values in [0, 1]).
    Returns a tensor of shape (B, C, H, W).
    """
    channel = x.size(1)
    window = _create_window(window_size, channel).to(x.device, dtype=x.dtype)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channel)
    mu_y = F.conv2d(y, window, padding=pad, groups=channel)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channel) - mu_xy

    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim


class L1SSIMLoss(nn.Module):
    """
    Combined L1 + SSIM loss widely used for MVTec-AD anomaly detection:
        loss = (1 - alpha) * L1 + alpha * (1 - SSIM)
    Default alpha = 0.5 balances absolute pixel differences and structural similarity.
    """
    def __init__(self, alpha: float = 0.5, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        ssim_val = compute_ssim_map(pred, target, self.window_size).mean()
        return (1.0 - self.alpha) * l1 + self.alpha * (1.0 - ssim_val)
