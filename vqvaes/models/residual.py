import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Passes input of size `in_channels` through 2 conv layers and adds an residual connection to input.

    Returns a tensor of the same size as the input.
    """

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
