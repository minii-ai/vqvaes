import torch
import torch.nn as nn
from einops import rearrange

from .residual import ResidualStack
from .vector_quantize import VectorQuantizeEMA


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
        scale_factor: int,
    ):
        assert scale_factor in [2, 4], "Scale factor must be 2 or 4."

        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_residual_channels = num_residual_channels

        if scale_factor == 2:
            layers = [
                nn.Conv2d(
                    in_channels, num_channels // 2, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    num_channels // 2, num_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        elif scale_factor == 4:
            layers = [
                nn.Conv2d(
                    in_channels,
                    num_channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    num_channels // 2,
                    num_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    num_channels, num_channels, kernel_size=3, stride=1, padding=1
                ),
            ]

        layers.append(
            ResidualStack(num_channels, num_residual_blocks, num_residual_channels)
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
        scale_factor: int,
    ):
        assert scale_factor in [2, 4], "Scale factor must be 2 or 4."

        super().__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(in_channels, num_residual_blocks, num_residual_channels),
        ]
        if scale_factor == 2:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                ]
            )
        elif scale_factor == 4:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        in_channels // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        in_channels // 2,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                ]
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class VQVAE2(nn.Module):
    """
    VQVAE 2 from Generating Diverse High-Fidelity Images with VQ-VAE-2 "https://arxiv.org/abs/1906.00446"
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.encoder_b = Encoder(
            in_channels,
            num_channels,
            num_residual_blocks,
            num_residual_channels,
            scale_factor=4,
        )
        self.encoder_t = Encoder(
            num_channels,
            num_channels,
            num_residual_blocks,
            num_residual_channels,
            scale_factor=2,
        )
        self.pre_vq_t = nn.Conv2d(
            num_channels, codebook_dim, kernel_size=1, stride=1, padding=0
        )
        self.pre_vq_b = nn.Conv2d(
            num_channels + codebook_dim,
            codebook_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.vq_b = VectorQuantizeEMA(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        self.vq_t = VectorQuantizeEMA(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        self.upsample = nn.ConvTranspose2d(
            codebook_dim, codebook_dim, kernel_size=4, stride=2, padding=1
        )

        self.decoder_t = Decoder(
            codebook_dim,
            codebook_dim,
            num_residual_blocks,
            num_residual_channels,
            scale_factor=2,
        )
        self.decoder = Decoder(
            codebook_dim * 2,
            in_channels,
            num_residual_blocks,
            num_residual_channels,
            scale_factor=4,
        )

    def quantize(self, x: torch.Tensor, level: str, is_training: bool = False):
        assert level in ["top", "bottom"]

        # rearrange shape so channel is last dim
        x = rearrange(x, "b c h w -> b h w c")
        if level == "top":
            vq_result = self.vq_t(x, is_training)
        else:
            vq_result = self.vq_b(x, is_training)

        # reshape back to bchw
        quantized = rearrange(vq_result["quantize"], "b h w c -> b c h w")
        result = {**vq_result, "quantize": quantized}

        return result

    def encode(self, x: torch.Tensor, is_training: bool = False):
        encoded_b = self.encoder_b(x)  # downscale by 4x
        # downscale by prev encoding by 2x (total 8x)
        encoded_t = self.encoder_t(encoded_b)

        # quantize the top level
        vq_top = self.quantize(
            self.pre_vq_t(encoded_t), level="top", is_training=is_training
        )

        # upsample top level quantization
        upsampled_vq_top = self.upsample(vq_top["quantize"])

        # concat bottom level with upsampled top level quantization
        encoded_b = torch.cat([encoded_b, upsampled_vq_top], dim=1)

        # quantize bottom level
        vq_bottom = self.quantize(
            self.pre_vq_b(encoded_b), level="bottom", is_training=is_training
        )

        return vq_bottom, vq_top

    def decode(self, quantized_bottom: torch.Tensor, quantized_top: torch.Tensor):
        # decode top level
        decoded_top = self.decoder_t(quantized_top)

        # concat quantized bottom with decoded top and decode
        x = torch.cat([quantized_bottom, decoded_top], dim=1)
        x = self.decoder(x)

        return x

    def forward(self, inputs: torch.Tensor, is_training: bool = False):
        vq_bottom, vq_top = self.encode(inputs, is_training)
        recon = self.decode(vq_bottom["quantize"], vq_top["quantize"])

        return recon, vq_bottom, vq_top
