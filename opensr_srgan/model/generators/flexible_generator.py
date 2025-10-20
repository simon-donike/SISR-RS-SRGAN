"""Flexible generator variants built from shared model blocks."""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn

from ..model_blocks import (
    ResidualBlockNoBN,
    RCAB,
    RRDB,
    LKAResBlock,
    make_upsampler,
)

__all__ = ["FlexibleGenerator", "flexible_generator"]


_BLOCK_REGISTRY: Dict[str, Callable[[int, int], nn.Module]] = {
    "res": lambda n_channels, kernel: ResidualBlockNoBN(
        n_channels=n_channels,
        kernel_size=kernel,
        res_scale=0.2,
    ),
    "rcab": lambda n_channels, kernel: RCAB(
        n_channels=n_channels,
        kernel_size=kernel,
        reduction=16,
        res_scale=0.2,
    ),
    "rrdb": lambda n_channels, kernel: RRDB(
        n_features=n_channels,
        growth_channels=max(16, n_channels // 3),
        res_scale=0.2,
    ),
    "lka": lambda n_channels, kernel: LKAResBlock(
        n_channels=n_channels,
        kernel_size=kernel,
        res_scale=0.2,
    ),
}


class FlexibleGenerator(nn.Module):
    """Drop-in SR generator supporting multiple residual block types."""

    def __init__(
        self,
        in_channels: int = 6,
        n_channels: int = 96,
        n_blocks: int = 32,
        small_kernel: int = 3,
        large_kernel: int = 9,
        scale: int = 8,
        block_type: str = "rcab",
    ) -> None:
        super().__init__()

        if scale not in {2, 4, 8}:
            raise ValueError("scale must be one of {2, 4, 8}")

        block_key = block_type.lower()
        if block_key not in _BLOCK_REGISTRY:
            raise ValueError("block_type must be one of {'res', 'rcab', 'rrdb', 'lka'}")

        self.scale = scale

        padding_large = large_kernel // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, large_kernel, padding=padding_large),
            nn.PReLU(),
        )

        block_factory = _BLOCK_REGISTRY[block_key]
        self.body = nn.Sequential(
            *[
                block_factory(n_channels, small_kernel)
                for _ in range(n_blocks)
            ]
        )
        self.body_tail = nn.Conv2d(n_channels, n_channels, small_kernel, padding=small_kernel // 2)
        self.upsampler = make_upsampler(n_channels, scale)
        self.tail = nn.Conv2d(n_channels, in_channels, large_kernel, padding=padding_large)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res
        feat = self.upsampler(feat)
        return self.tail(feat)


flexible_generator = FlexibleGenerator
