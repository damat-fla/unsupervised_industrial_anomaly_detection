import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

from engine import compute_anomaly_map


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


def visualize_anomalies(model, dataloader, device, num_samples=4):
    """
    Plot Input, Reconstruction, Ground Truth, and Anomaly Map side by side for test samples.
    """
    model.eval()
    batch = next(iter(dataloader))
    images = batch["image"][:num_samples].to(device)
    masks = batch["mask"][:num_samples]
    defect_types = batch["defect_type"][:num_samples]

    with torch.no_grad():
        reconstructions = model(images)
        anomaly_maps = compute_anomaly_map(images, reconstructions)

    images = images.cpu()
    reconstructions = reconstructions.cpu()
    anomaly_maps = anomaly_maps.cpu()

    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    titles = ["Input", "Reconstruction", "Ground Truth", "Anomaly Map"]
    for i in range(num_samples):
        axes[i][0].imshow(images[i].permute(1, 2, 0).numpy())
        axes[i][0].set_ylabel(defect_types[i], fontsize=12)
        axes[i][1].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[i][2].imshow(masks[i, 0].numpy(), cmap="gray")
        im = axes[i][3].imshow(anomaly_maps[i, 0].numpy(), cmap="jet")
        plt.colorbar(im, ax=axes[i][3], fraction=0.046, pad=0.04)

        if i == 0:
            for j in range(4):
                axes[0][j].set_title(titles[j])

        for j in range(4):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    plt.tight_layout()
    plt.show()

