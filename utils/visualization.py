import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils



def visualize_batch(batch, max_images=16, nrow=4, title="Training Batch", figsize=(8, 8)):
    """
    Visualize a grid of images from a batch dictionary, tensor, or DataLoader.
    """
    if hasattr(batch, "__iter__") and not isinstance(batch, (dict, torch.Tensor)):
        batch = next(iter(batch))
    images = batch["image"] if isinstance(batch, dict) else batch

    grid = vutils.make_grid(images[:max_images], nrow=nrow, padding=2)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_anomalies(model, dataloader, device, num_samples=4, visualize_reconstruction=False):
    """
    Plot Input Image, Ground Truth, Anomaly Heatmap, Heatmap Overlay,
    and optionally Reconstruction for test samples.
    """
    model.eval()
    batch = next(iter(dataloader))
    images = batch["image"][:num_samples].to(device)
    masks = batch["mask"][:num_samples]
    defect_types = batch["defect_type"][:num_samples]
    image_names = batch["path"][:num_samples]

    with torch.no_grad():
        recon, target = model(images)
        anomaly_maps = model.compute_anomaly_map(recon, target)

    images = images.cpu()
    masks = masks.cpu()
    anomaly_maps = anomaly_maps.cpu()
    recon = recon.cpu()

    titles = ["Input Image", "Ground Truth", "Anomaly Map", "Overlay"]
    if visualize_reconstruction:
        titles.append("Reconstruction")

    ncols = len(titles)
    fig, axes = plt.subplots(num_samples, ncols, figsize=(3.5 * ncols, 3.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 1. Immagine RGB originale [H, W, 3] normalizzata in [0, 1]
        img_rgb = images[i].permute(1, 2, 0).numpy()
        img_rgb = np.clip(img_rgb, 0.0, 1.0)
        axes[i][0].imshow(img_rgb)
        axes[i][0].set_ylabel(f"{defect_types[i]}\n{image_names[i].split('\\')[-1]}", fontsize=11, fontweight="bold")

        # 2. Ground Truth Mask [H, W]
        axes[i][1].imshow(masks[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)

        # 3. Anomaly Heatmap [H, W]
        heat_map = anomaly_maps[i, 0].numpy()
        im = axes[i][2].imshow(heat_map, cmap="jet")
        plt.colorbar(im, ax=axes[i][2], fraction=0.046, pad=0.04)

        # 4. Overlay: Heatmap sovrapposta con trasparenza (alpha=0.45) sull'immagine RGB
        axes[i][3].imshow(img_rgb)
        axes[i][3].imshow(heat_map, cmap="jet", alpha=0.45)

        # 5. Reconstruction (opzionale)
        if visualize_reconstruction:
            recon_rgb = np.clip(recon[i].permute(1, 2, 0).numpy(), 0.0, 1.0)
            axes[i][4].imshow(recon_rgb)

        # Titoli solo sulla prima riga
        if i == 0:
            for j in range(ncols):
                axes[0][j].set_title(titles[j], fontsize=12, pad=8)

        # Rimuove i tick degli assi per una visualizzazione pulita
        for j in range(ncols):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    plt.tight_layout()
    plt.show()

