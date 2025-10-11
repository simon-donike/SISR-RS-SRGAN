"""Compatibility layer for flexible generator architectures."""

from __future__ import annotations

from .flexible_generator import FlexibleGenerator, flexible_generator
from ..model_blocks import (
    ConvolutionalBlock,
    SubPixelConvolutionalBlock,
    ResidualBlock,
    ResidualBlockNoBN,
    RCAB,
    DenseBlock5,
    RRDB,
    LKA,
    LKAResBlock,
    make_upsampler,
)

__all__ = [
    "FlexibleGenerator",
    "flexible_generator",
    "ConvolutionalBlock",
    "SubPixelConvolutionalBlock",
    "ResidualBlock",
    "ResidualBlockNoBN",
    "RCAB",
    "DenseBlock5",
    "RRDB",
    "LKA",
    "LKAResBlock",
    "make_upsampler",
]
