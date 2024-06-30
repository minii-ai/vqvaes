import torch
import torch.nn as nn

from .layers import Decoder, Encoder
from .vq import VQ


class VQVAE(nn.Module):
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

    def encode(self):
        pass

    def decode(self):
        pass

    def forward(self, inputs: torch.Tensor):
        pass
