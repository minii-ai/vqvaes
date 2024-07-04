from typing import Union

import torch
import torch.nn as nn
from einops import rearrange

from .layers import Decoder, Encoder
from .vector_quantize import VectorQuantize, VectorQuantizeEMA


class VQVAE(nn.Module):
    """
    VQVAE from Neural Discrete Representation Learning "https://arxiv.org/abs/1711.00937"
    """

    def __init__(
        self,
        encoder: Encoder,
        vq: Union[VectorQuantize, VectorQuantizeEMA],
        decoder: Decoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.pre_vq = nn.Conv2d(
            encoder.num_channels, vq.codebook_dim, kernel_size=1, stride=1, padding=0
        )
        self.vq = vq
        self.decoder = decoder

    @property
    def config(self):
        use_ema = isinstance(self.vq, VectorQuantizeEMA)
        decay = self.vq.decay if use_ema else None

        return {
            "in_channels": self.encoder.in_channels,
            "num_channels": self.encoder.num_channels,
            "num_residual_blocks": self.encoder.num_residual_blocks,
            "num_residual_channels": self.encoder.num_residual_channels,
            "codebook_size": self.vq.codebook_size,
            "codebook_dim": self.vq.codebook_dim,
            "commitment_cost": self.vq.commitment_cost,
            "use_ema": use_ema,
            "decay": decay,
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
