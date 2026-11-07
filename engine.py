import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from anomalib.metrics import AUROC, AUPRO
from anomalib.data import ImageBatch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred, target = model(images)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Val", leave=False)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            pred, target = model(images)
            loss = criterion(pred, target)

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

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

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device) if val_loader else train_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs_trained"] = epoch

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            history["best_epoch"] = best_epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "checkpoints" / "best_model.pth")
        else:
            patience_counter += 1

        torch.save(model.state_dict(), save_path / "checkpoints" / f"last_model.pth")

        with open(save_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if patience and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return history


def evaluate(model, test_loader, device, top_k_ratio=0.01, save_dir="./experiments/baseline"):
    model.eval()

    image_auroc = AUROC(fields=["pred_score", "gt_label"])
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.3)

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = (batch["mask"] > 0.5).to(device)
            labels = batch["label"].to(device).int()

            pred, target = model(images)
            anomaly_maps = model.compute_anomaly_map(pred, target)

            # Robust image score using top-k average anomaly pixels instead of pure max
            flat_maps = anomaly_maps.flatten(1)
            k = max(1, int(flat_maps.size(1) * top_k_ratio))
            image_scores = torch.topk(flat_maps, k=k, dim=1).values.mean(dim=1)

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

    metrics = {
        "image_auroc": image_auroc.compute().item(),
        "pixel_auroc": pixel_auroc.compute().item(),
        "aupro": aupro.compute().item(),
    }

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
