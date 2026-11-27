import os
import random
import numpy as np
import torch
from torchvision.transforms.v2.functional import gaussian_blur


def set_seed(seed: int = 42):
    """
    Fix all sources of randomness for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}.")


def get_device(device_name: str = None):
    """
    If a device name is provided, it will be used. Otherwise, automatically select the device.
    """
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def smooth_anomaly_map(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Gaussian smoothing of an anomaly map [B, 1, H, W] (kernel size 2*round(2*sigma)+1).
    sigma = 0 (or None) returns the map unchanged.
    """
    if not sigma:
        return anomaly_map
    kernel_size = int(2 * round(2 * sigma) + 1)
    return gaussian_blur(anomaly_map, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
