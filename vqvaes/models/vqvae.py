import json

import torch
import torch.nn as nn
from einops import rearrange

from .residual import ResidualStack
from .vector_quantize import VectorQuantize, VectorQuantizeEMA


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_residual_channels = num_residual_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, num_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                num_channels // 2, num_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_channels, num_residual_blocks, num_residual_channels),
        )

    def forward(self, inputs: torch.Tensor):
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        out_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_channels, num_residual_blocks, num_residual_channels),
            nn.ConvTranspose2d(
                num_channels, num_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_channels // 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class VQVAE(nn.Module):
    """
    VQVAE from Neural Discrete Representation Learning "https://arxiv.org/abs/1711.00937"
    """

    @staticmethod
    def load_from_checkpoint(config_path: str, weights_path: str):
        with open(config_path, "r") as f:
            model_config = json.load(f)
        model = VQVAE(**model_config)
        model.load_state_dict(torch.load(weights_path))

        return model

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        decay: float = 0.99,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_residual_channels = num_residual_channels
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.decay = decay

        self.encoder = Encoder(
            in_channels=in_channels,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks,
            num_residual_channels=num_residual_channels,
        )
        self.pre_vq = nn.Conv2d(
            num_channels, codebook_dim, kernel_size=1, stride=1, padding=0
        )

        if use_ema:
            vq = VectorQuantizeEMA(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            vq = VectorQuantize(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                commitment_cost=commitment_cost,
            )

        self.vq = vq
        self.decoder = Decoder(
            in_channels=codebook_dim,
            num_channels=num_channels,
            out_channels=in_channels,
            num_residual_blocks=num_residual_blocks,
            num_residual_channels=num_residual_channels,
        )

    @property
    def config(self):

        return {
            "in_channels": self.in_channels,
            "num_channels": self.encoder.num_channels,
            "num_residual_blocks": self.num_residual_blocks,
            "num_residual_channels": self.num_residual_channels,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "commitment_cost": self.commitment_cost,
            "use_ema": self.use_ema,
            "decay": self.decay,
        }

    def encode(self, inputs: torch.Tensor, is_training: bool = False):
        x = self.encoder(inputs)
        x = self.pre_vq(x)

        # pass encoder output thr. vq layer
        x = rearrange(x, "b c h w -> b h w c")
        vq_result = self.vq(x, is_training)
        quantized = rearrange(vq_result["quantize"], "b h w c -> b c h w")
        result = {**vq_result, "quantize": quantized}

        return result

    def decode(self, quantized: torch.Tensor):
        return self.decoder(quantized)

    def forward(self, inputs: torch.Tensor, is_training: bool = False):
        vq_result = self.encode(inputs, is_training)
        quantized = vq_result["quantize"]
        recon = self.decode(quantized)

        return recon, vq_result
