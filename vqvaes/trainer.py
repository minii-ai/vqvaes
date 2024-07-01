import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .evals import evaluate_vqvae, evaluate_vqvae_recon
from .models import VQVAE


class VQVAETrainer:
    def __init__(
        self,
        vqvae: VQVAE,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_eval_indices: list[int] = [0],
        test_eval_indices: list[int] = [0],
        lr: float = 1e-4,
        iterations: int = 100000,
        weight_decay: float = 0.0,
        checkpoint_every: int = 1000,
        save_dir: str = None,
        device: str = "cpu",
    ):
        self.vqvae = vqvae
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_eval_indices = train_eval_indices
        self.test_eval_indices = test_eval_indices
        self.lr = lr
        self.iterations = iterations
        self.weight_decay = weight_decay
        self.checkpoint_every = checkpoint_every
        self.save_dir = save_dir
        self.device = device
        self.writer = None

        if self.save_dir:
            self.writer = SummaryWriter(self.save_dir)

    def train(self):
        print("[INFO] Training...")
        num_params = sum(p.numel() for p in self.vqvae.parameters())
        print(f"[INFO] Number of parameters: {num_params}")

        if self.save_dir:
            # save model config
            config_path = os.path.join(self.save_dir, "config.json")
            model_config = {
                "in_channels": 3,
                "num_channels": self.vqvae.num_channels,
                "num_residual_blocks": self.vqvae.num_residual_blocks,
                "num_residual_channels": self.vqvae.num_residual_channels,
                "codebook_size": self.vqvae.codebook_size,
                "codebook_dim": self.vqvae.codebook_dim,
            }

            with open(config_path, "w") as f:
                json.dump(model_config, f)

        vqvae = self.vqvae
        vqvae.to(self.device)

        optimizer = torch.optim.Adam(
            vqvae.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        i = 0
        train_losses = []
        with tqdm(total=self.iterations, position=0) as pbar:
            while True:
                for data in self.train_loader:
                    vqvae.train()
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

                    if self.writer:
                        self.writer.add_scalars("loss", {"train": loss.item()}, i)
                        self.writer.add_scalars(
                            "perplexity", {"train": perplexity.item()}, i
                        )
                        self.writer.add_scalars(
                            "recon_loss", {"train": recon_loss.item()}, i
                        )

                    if (i + 1) % self.checkpoint_every == 0 or i == self.iterations - 1:
                        # save weights
                        if self.save_dir:
                            save_path = os.path.join(self.save_dir, "weights.pt")
                            torch.save(vqvae.state_dict(), save_path)

                        if self.writer:
                            test_evals = evaluate_vqvae(self.test_loader, vqvae)

                            self.writer.add_scalars(
                                "loss", {"test": test_evals["loss"]}, i
                            )
                            self.writer.add_scalars(
                                "perplexity", {"test": test_evals["perplexity"]}, i
                            )
                            self.writer.add_scalars(
                                "recon_loss", {"test": test_evals["recon_loss"]}, i
                            )

                            # compare original vs recon images
                            train_comparison = evaluate_vqvae_recon(
                                self.train_loader.dataset,
                                indices=self.train_eval_indices,
                                model=vqvae,
                            )
                            test_comparison = evaluate_vqvae_recon(
                                self.test_loader.dataset,
                                indices=self.test_eval_indices,
                                model=vqvae,
                            )

                            self.writer.add_image(
                                "train/eval",
                                np.array(train_comparison).transpose(2, 0, 1),
                                i,
                            )
                            self.writer.add_image(
                                "test/eval",
                                np.array(test_comparison).transpose(2, 0, 1),
                                i,
                            )

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
