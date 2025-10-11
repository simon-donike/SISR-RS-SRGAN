"""SRGAN discriminator architectures built from shared blocks."""

from __future__ import annotations

import torch
from torch import nn

from .model_blocks import ConvolutionalBlock

__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    """Standard SRGAN discriminator as defined in the original paper."""

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 8,
        fc_size: int = 1024,
    ) -> None:
        super().__init__()

        conv_blocks: list[nn.Module] = []
        current_in = in_channels
        out_channels = n_channels
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else current_in * 2) if i % 2 == 0 else current_in
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

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch_size = imgs.size(0)
        feats = self.conv_blocks(imgs)
        pooled = self.adaptive_pool(feats)
        flat = pooled.view(batch_size, -1)
        hidden = self.leaky_relu(self.fc1(flat))
        return self.fc2(hidden)
