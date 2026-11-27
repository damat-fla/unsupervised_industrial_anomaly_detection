import csv
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from anomalib.metrics import AUROC, AUPRO
from anomalib.data import ImageBatch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    # Pay attention when using model.train(). It recursively sets all submodules to training mode.
    # Since my model contains the teacher (which I want to be frozen), I should ensure that it is 
    # not affected by model.train(). To do so, it suffices to override the train() method of the 
    # model to keep the teacher in eval mode.

    total_loss = 0.0
    n_samples = 0  # samples actually processed (drop_last=True may skip the last incomplete batch)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred, target = model(images)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / n_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            pred, target = model(images)
            loss = criterion(pred, target)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def train(model, train_loader, criterion, optimizer, epochs, device, val_loader=None, save_dir="./experiments/baseline", patience=None):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create a directory for checkpoints
    (save_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "epochs_trained": 0, "best_epoch": 1}

    pbar = tqdm(range(1, epochs + 1), desc="Training")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device) if val_loader else train_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs_trained"] = epoch

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "checkpoints" / "best_model.pth")
        else:
            patience_counter += 1

        torch.save(model.state_dict(), save_path / "checkpoints" / "last_model.pth")

        with open(save_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if patience and patience_counter >= patience:
            pbar.write(f"Early stopping triggered at epoch {epoch}")
            break

    return history


def evaluate(model, test_loader, device, top_k_ratio=0.01, save_dir="./experiments/baseline", suffix=""):
    """
    Computes the test metrics and, if save_dir is given, saves metrics{suffix}.json and image_scores{suffix}.csv
    (suffix="_sigma4" when re-evaluating a checkpoint, so that the original results are not overwritten).

    Two kinds of models are supported:
    - custom models (feature/pixel reconstruction): pred, target = model(images), then model.compute_anomaly_map.
      The image score is the mean of the top-k_ratio pixels of the anomaly map.
    - methods from the literature (models/literature_models.py), recognized by a predict() method returning the
      anomaly map and the native image score of the method. The native score is the main score ("score",
      "image_auroc"); the top-k score is also saved ("score_top1pct", "image_auroc_top1pct") to compare the
      methods with the same aggregation of the map.
    """
    model.eval()
    has_native_score = hasattr(model, "predict")

    image_auroc = AUROC(fields=["pred_score", "gt_label"])
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.3)
    if has_native_score:
        image_auroc_top1pct = AUROC(fields=["pred_score", "gt_label"])

    # Per-image anomaly scores, used for the error analysis (e.g. AUROC per defect type)
    image_records = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating", leave=True):
            images = batch["image"].to(device)
            masks = (batch["mask"] > 0.5).to(device)
            labels = batch["label"].to(device).int()

            if has_native_score:
                anomaly_maps, native_scores = model.predict(images)
            else:
                pred, target = model(images)
                anomaly_maps = model.compute_anomaly_map(pred, target)

            # Robust image score using top-k average anomaly pixels instead of pure max
            flat_maps = anomaly_maps.flatten(1)
            k = max(1, int(flat_maps.size(1) * top_k_ratio))
            topk_scores = torch.topk(flat_maps, k=k, dim=1).values.mean(dim=1)

            image_scores = native_scores if has_native_score else topk_scores

            metric_batch = ImageBatch(
                image=images,
                pred_score=image_scores,
                gt_label=labels,
                anomaly_map=anomaly_maps,
                gt_mask=masks,
            )

            image_auroc.update(metric_batch)
            pixel_auroc.update(metric_batch)
            aupro.update(metric_batch)
            if has_native_score:
                image_auroc_top1pct.update(ImageBatch(image=images, pred_score=topk_scores, gt_label=labels))

            for i, (path, defect_type, label, score) in enumerate(zip(
                batch["path"], batch["defect_type"], labels.tolist(), image_scores.tolist()
            )):
                record = {"path": path, "defect_type": defect_type, "label": label, "score": score}
                if has_native_score:
                    record["score_top1pct"] = topk_scores[i].item()
                image_records.append(record)

    metrics = {
        "image_auroc": image_auroc.compute().item(),
        "pixel_auroc": pixel_auroc.compute().item(),
        "aupro": aupro.compute().item(),
    }
    if has_native_score:
        metrics["image_auroc_top1pct"] = image_auroc_top1pct.compute().item()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / f"metrics{suffix}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(save_path / f"image_scores{suffix}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_records[0].keys()))
            writer.writeheader()
            writer.writerows(image_records)

    return metrics
