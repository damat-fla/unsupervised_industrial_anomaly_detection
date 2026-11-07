import os
import random
from pathlib import Path
import numpy as np
import torch


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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device = None, strict: bool = True):
    """
    Load weights from a checkpoint file (.pth) into the provided model.
    Handles both raw state_dicts and checkpoint dictionaries containing 'model_state_dict' or 'state_dict'.

    Parameters:
    -----------
    - model: torch.nn.Module, the target model instance.
    - checkpoint_path: str or Path, path to the .pth checkpoint file.
    - device: torch.device, optional device to map the weights to (e.g. cpu or cuda).
    - strict: bool, whether to strictly enforce that the keys in state_dict match model.state_dict().

    Returns:
    --------
    - model: the model with loaded weights (in eval mode by default).
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file}")

    map_location = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=False)

    # Check if checkpoint is a dictionary wrapper or just the state_dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    model.to(map_location)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_file}")
    return model
