import torch
import torch.nn.functional as F


def vqvae_loss(recon: torch.Tensor, vq_result: dict, images: torch.Tensor):
    recon_loss = F.mse_loss(recon, images)
    loss = recon_loss + vq_result["loss"]

    return loss
