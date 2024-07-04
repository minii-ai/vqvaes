import torch
import torch.nn as nn

from .residual import ResidualStack


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
    pass
