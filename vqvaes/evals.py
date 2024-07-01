import io

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from .loss import vqvae_loss
from .models import VQVAE


def evaluate_vqvae(dataloader: DataLoader, model: VQVAE):
    """
    Evaluates VQVAE's loss and perplexity over a dataloader
    """
    loss = 0
    recon_loss = 0
    perplexity = 0

    device = next(model.parameters())
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating VQVAE"):
            images = batch["images"].to(device)
            recon, vq_result = model(images)
            batch_recon_loss, batch_loss = vqvae_loss(recon, vq_result, images)
            batch_perplexity = vq_result["perplexity"]

            loss += batch_loss.item()
            recon_loss += batch_recon_loss.item()
            perplexity += batch_perplexity.item()

    loss /= len(dataloader)
    perplexity /= len(dataloader)

    return {"loss": loss, "recon_loss": recon_loss, "perplexity": perplexity}


def figure_to_image(figure):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    return image


def evaluate_vqvae_recon(
    dataset: Dataset, indices: list[int], model: VQVAE, normalize: bool = True
):
    """
    Evaluates VQVAE by comparing original images to their reconstructions.
    """

    device = next(model.parameters()).device
    model.eval()

    images = [dataset[i]["image"] for i in indices]
    images = torch.stack(images).to(device)

    with torch.no_grad():
        recon, _ = model(images)

    grid_size = int(len(indices) ** 0.5)
    orig_grid = make_grid(
        images.cpu(), nrow=grid_size, value_range=(-1, 1), normalize=normalize
    )
    recon_grid = make_grid(
        recon.cpu(), nrow=grid_size, value_range=(-1, 1), normalize=normalize
    )

    # Combine the grids horizontally
    combined_grid = torch.cat([orig_grid, recon_grid], dim=2)

    # Create figure and add a single subplot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot combined grid
    ax.imshow(combined_grid.permute(1, 2, 0))
    ax.axis("off")

    # Add titles
    ax.text(
        0.25,
        1.02,
        "Original",
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.text(
        0.75,
        1.02,
        "Reconstructed",
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )

    plt.tight_layout()
    return figure_to_image(fig)
