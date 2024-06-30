import torch
import torch.nn as nn
from einops import rearrange

from .layers import Decoder, Encoder
from .vq import VQ


class VQVAE(nn.Module):
    """
    VQVAE from Neural Discrete Representation Learning "https://arxiv.org/abs/1711.00937"
    """

    def __init__(
        self,
        encoder: Encoder,
        vq: VQ,
        decoder: Decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.pre_vq = nn.Conv2d(
            encoder.num_channels, vq.codebook_dim, kernel_size=1, stride=1, padding=0
        )
        self.vq = vq
        self.decoder = decoder

    def encode(self, inputs: torch.Tensor):
        x = self.encoder(inputs)
        x = self.pre_vq(x)

        # pass encoder output thr. vq layer
        x = rearrange(x, "b c h w -> b h w c")
        vq_result = self.vq(x)
        quantized = rearrange(vq_result["quantized"], "b h w c -> b c h w")
        result = {**vq_result, "quantized": quantized}

        return result

    def decode(self, quantized: torch.Tensor):
        return self.decoder(quantized)

    def forward(self, inputs: torch.Tensor):
        vq_result = self.encode(inputs)
        quantized = vq_result["quantized"]
        recon = self.decode(quantized)

        return recon, vq_result
