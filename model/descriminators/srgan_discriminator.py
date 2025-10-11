"""SRGAN discriminator architectures built from shared blocks."""

from __future__ import annotations

import torch
from torch import nn

from ..model_blocks import ConvolutionalBlock

__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    """Standard SRGAN discriminator as defined in the original paper."""

    def __init__(
        self,
        in_channels: int = 3,
        n_blocks: int = 8,
    ) -> None:
        super().__init__()

        if n_blocks < 1:
            raise ValueError("The SRGAN discriminator requires at least one block.")

        kernel_size = 3
        base_channels = 64
        fc_size = 1024

        conv_blocks: list[nn.Module] = []
        current_in = in_channels
        out_channels = base_channels
        for i in range(n_blocks):
            if i == 0:
                out_channels = base_channels
            elif i % 2 == 0:
                out_channels = current_in * 2
            else:
                out_channels = current_in

            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    batch_norm=i != 0,
                    activation="LeakyReLu",
                )
            )
            current_in = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)

        self.base_channels = base_channels
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.n_blocks = n_blocks

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch_size = imgs.size(0)
        feats = self.conv_blocks(imgs)
        pooled = self.adaptive_pool(feats)
        flat = pooled.view(batch_size, -1)
        hidden = self.leaky_relu(self.fc1(flat))
        return self.fc2(hidden)
