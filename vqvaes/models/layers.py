import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        return x + self.conv(x)


class ResidualStack(nn.Module):
    """
    A list of ResidualBlocks
    """

    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
    ):
        super().__init__()
        blocks = []

        for i in range(num_residual_blocks):
            blocks.append(ResidualBlock(in_channels, num_residual_channels))

        self.stack = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.stack(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
    ):
        super().__init__()
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
        out_channels: int,
        num_residual_blocks: int,
        num_residual_channels: int,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(in_channels, num_residual_blocks, num_residual_channels),
            nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels // 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)
