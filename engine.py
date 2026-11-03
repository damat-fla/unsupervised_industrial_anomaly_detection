'''
I want this class to contain functions for training and testing a PyTorch model on the MVTec 3D-AD dataset.
'''

import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from anomalib.metrics import AUROC, AUPRO
from anomalib.data import ImageBatch

from losses import compute_ssim_map


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
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
            outputs = model(images)
            loss = criterion(outputs, images)

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader.dataset)



def train(model, train_loader, criterion, optimizer, epochs, device,val_loader = None, save_dir="./experiments/baseline"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path / "best_model.pth")

        torch.save(model.state_dict(), save_path / "last_model.pth")

        with open(save_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history


def compute_anomaly_map(images, reconstructions, method="l1_ssim", alpha=0.5):
    if method == "l1_ssim":
        l1_map = torch.mean(torch.abs(images - reconstructions), dim=1, keepdim=True)
        ssim = torch.mean(compute_ssim_map(images, reconstructions), dim=1, keepdim=True)
        return (1.0 - alpha) * l1_map + alpha * (1.0 - ssim)
    return torch.mean((images - reconstructions) ** 2, dim=1, keepdim=True)


def evaluate(model, test_loader, device):
    model.eval()

    image_auroc = AUROC(fields=["pred_score", "gt_label"])
    pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"])
    aupro = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.3)

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = (batch["mask"] > 0.5).to(device)
            labels = batch["label"].to(device).int()

            reconstructions = model(images)
            anomaly_maps = compute_anomaly_map(images, reconstructions)

            image_scores = anomaly_maps.flatten(1).max(dim=1).values

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

    return {
        "image_auroc": image_auroc.compute().item(),
        "pixel_auroc": pixel_auroc.compute().item(),
        "aupro": aupro.compute().item(),
    }