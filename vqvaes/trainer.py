import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import VQVAE


class VQVAETrainer:
    def __init__(
        self,
        vqvae: VQVAE,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 1e-4,
        iterations: int = 100000,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        self.vqvae = vqvae
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.device = device

    def train(self):
        vqvae = self.vqvae
        vqvae.to(self.device)
        vqvae.train()

        optimizer = torch.optim.Adam(
            vqvae.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        i = 0
        train_losses = []
        with tqdm(total=self.iterations) as pbar:
            while True:
                for data in self.train_loader:
                    images = data["images"].to(self.device)
                    recon, vq_result = vqvae(images)
                    perplexity = vq_result["perplexity"]

                    # calculate loss
                    recon_loss = F.mse_loss(recon, images)
                    loss = recon_loss + vq_result["loss"]
                    train_losses.append(loss.item())

                    # backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    i += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                        perplexity=perplexity.item(),
                        recon_loss=recon_loss.item(),
                        vq_loss=vq_result["loss"].item(),
                    )

                    if i >= self.iterations:
                        break

                if i >= self.iterations:
                    break
