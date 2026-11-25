from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class MVTec2DDataset(Dataset):
    """
    PyTorch Dataset for 2D MVTec AD images and ground truth defect masks.
    Supports any class under root/[class].
    """
    def __init__(self, root="./data/MVTec", class_name="bottle", split="train", transforms=None, target_transforms=None, cache=False):
        base_dir = Path(root) / class_name
        self.root = base_dir / class_name if (base_dir / class_name).is_dir() else base_dir
        self.split = split
        self.transform = transforms
        self.target_transform = target_transforms or transforms
        self.samples = self._load_samples()

        self.cache = cache
        self._cached_data = {} if cache else None

    def _load_samples(self):
        samples = []
        split_dir = self.root / self.split
        gt_root = self.root / "ground_truth"

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted(defect_dir.glob("*.png")):
                gt_path = gt_root / defect_type / f"{img_path.stem}_mask{img_path.suffix}"
                if not gt_path.exists():
                    gt_path = gt_root / defect_type / img_path.name

                samples.append({
                    "rgb_path": img_path,
                    "gt_path": gt_path if gt_path.exists() else None,
                    "label": label,
                    "defect_type": defect_type
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if self.cache and idx in self._cached_data:
            return self._cached_data[idx]
        
        item = self.samples[idx]
        image = Image.open(item["rgb_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if item["gt_path"]:
            mask = Image.open(item["gt_path"]).convert("L")
            mask = (self.target_transform(mask) > 0).float() if self.target_transform else mask
        else:
            mask = torch.zeros_like(image[:1])

        sample = {
            "image": image,
            "mask": mask,
            "label": item["label"],
            "defect_type": item["defect_type"],
            "path": str(item["rgb_path"])
        }

        if self.cache:
            self._cached_data[idx] = sample

        return sample
