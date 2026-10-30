'''
I want this class to contain functions for training and testing a PyTorch model on the MVTec 3D-AD dataset.
'''

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm


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


def compute_anomaly_map(images, reconstructions):
    return torch.mean((images - reconstructions) ** 2, dim=1, keepdim=True)


def evaluate(model, test_loader, device):
    model.eval()
    all_image_scores, all_image_labels = [], []
    all_pixel_scores, all_pixel_masks = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            reconstructions = model(images)
            anomaly_maps = compute_anomaly_map(images, reconstructions)
            image_scores = anomaly_maps.view(images.size(0), -1).max(dim=1)[0]

            all_image_scores.append(image_scores.cpu().numpy())
            all_image_labels.append(labels.numpy())
            all_pixel_scores.append(anomaly_maps.cpu().numpy().ravel())
            all_pixel_masks.append(masks.numpy().ravel())

    image_scores = np.concatenate(all_image_scores)
    image_labels = np.concatenate(all_image_labels)
    pixel_scores = np.concatenate(all_pixel_scores)
    pixel_masks = np.concatenate(all_pixel_masks).astype(int)

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_auroc = roc_auc_score(pixel_masks, pixel_scores)

    print(f"Evaluation -> Image AUROC: {image_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f}")
    return {"image_auroc": float(image_auroc), "pixel_auroc": float(pixel_auroc)}