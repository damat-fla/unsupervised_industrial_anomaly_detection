from pathlib import Path
import random
import shutil
from typing import Dict, Optional, Union


def _split_single_class(class_dir: Path, val_ratio: float, seed: int, force: bool) -> Dict[str, int]:
    train_dir = class_dir / "train"
    train_orig_dir = class_dir / "train_original"
    val_dir = class_dir / "validation"

    # Backup original train directory if not already done
    if not train_orig_dir.exists():
        if not train_dir.exists():
            raise FileNotFoundError(f"Neither 'train_original' nor 'train' found in {class_dir}")
        train_dir.rename(train_orig_dir)
    elif not force and (train_dir / "good").exists() and (val_dir / "good").exists():
        train_count = len(list((train_dir / "good").glob("*.png")))
        val_count = len(list((val_dir / "good").glob("*.png")))
        print(f"[{class_dir.name}] Already split (Train: {train_count}, Val: {val_count}). Use force=True to re-split.")
        return {"train": train_count, "val": val_count}

    src_good_dir = train_orig_dir / "good"
    if not src_good_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_good_dir}")

    images = sorted(src_good_dir.glob("*.png"))
    if not images:
        raise ValueError(f"No PNG images found in {src_good_dir}")

    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_files = shuffled[:val_size]
    train_files = shuffled[val_size:]

    dst_train_good = train_dir / "good"
    dst_val_good = val_dir / "good"

    if dst_train_good.exists():
        shutil.rmtree(dst_train_good)
    if dst_val_good.exists():
        shutil.rmtree(dst_val_good)

    dst_train_good.mkdir(parents=True, exist_ok=True)
    dst_val_good.mkdir(parents=True, exist_ok=True)

    for img_path in train_files:
        shutil.copy2(img_path, dst_train_good / img_path.name)

    for img_path in val_files:
        shutil.copy2(img_path, dst_val_good / img_path.name)

    print(f"[{class_dir.name}] Split complete -> Train: {len(train_files)}, Val: {len(val_files)}")
    return {"train": len(train_files), "val": len(val_files)}


def split_data(
    root: Union[str, Path] = "./data/MVTec",
    class_name: Optional[str] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Split training images of MVTec 2D into train and validation sets.
    Preserves original data in data/MVTec/[class]/train_original.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")

    if class_name is not None:
        target_dirs = [root_path / class_name]
    else:
        target_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    results = {}
    for c_dir in target_dirs:
        results[c_dir.name] = _split_single_class(c_dir, val_ratio, seed, force)

    return results
