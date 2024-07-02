import json

import torch

from .layers import Decoder, Encoder
from .vq import VQ
from .vqvae import VQVAE


def load_vqvae(config_path: str, weights_path: str) -> VQVAE:
    with open(config_path, "r") as f:
        model_config = json.load(f)

    model = build_vqvae(**model_config)
    model.load_state_dict(torch.load(weights_path))

    return model


def build_vqvae(
    in_channels: int,
    num_channels: int,
    num_residual_blocks: int,
    num_residual_channels: int,
    codebook_size: int,
    codebook_dim: int,
) -> VQVAE:
    encoder = Encoder(
        in_channels=in_channels,
        num_channels=num_channels,
        num_residual_blocks=num_residual_blocks,
        num_residual_channels=num_residual_channels,
    )

    vq = VQ(
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    decoder = Decoder(
        in_channels=codebook_dim,
        out_channels=in_channels,
        num_residual_blocks=num_residual_blocks,
        num_residual_channels=num_residual_channels,
    )

    return VQVAE(encoder=encoder, vq=vq, decoder=decoder)
