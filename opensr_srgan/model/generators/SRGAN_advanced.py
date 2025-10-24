"""Compatibility layer for flexible generator architectures.

This module provides a *stable import surface* that re-exports the flexible SR
generator and its building blocks from their canonical locations. It allows
downstream code and docs to import everything related to generator assembly
from a single place, while we retain freedom to refactor internal package
structure.

Exports:
    - ``FlexibleGenerator`` / ``flexible_generator``: A modular SR backbone that
      can be instantiated with different residual block families (``res``,
      ``rcab``, ``rrdb``, ``lka``). See ``flexible_generator.FlexibleGenerator``.
    - Core blocks used by the generator (``ResidualBlockNoBN``, ``RCAB``,
      ``RRDB``, ``LKAResBlock``) and utilities (``make_upsampler``, etc.).

How it works:
    - ``FlexibleGenerator`` accepts a ``block_type`` string and assembles the body
      from the corresponding residual block implementation. The upsampling stage
      is created via ``make_upsampler`` (typically PixelShuffle-based), and the
      head/tail use configurable kernel sizes.
    - All concrete block classes are re-exported so callers can build custom
      backbones or plug individual blocks into their own networks without
      depending on internal file paths.

Why this layer exists:
    - To keep *import paths stable* across refactors:
        ``from opensr_srgan.model.generators import FlexibleGenerator``
      remains valid even if the underlying files move.
    - To simplify documentation and API discovery: one module lists the generator
      options and the available building blocks.

Examples:
    >>> from opensr_srgan.model.generators import FlexibleGenerator
    >>> g = FlexibleGenerator(in_channels=3, block_type="rcab", scale=4)
    >>> y = g(torch.randn(1, 3, 64, 64))  # -> (1, 3, 256, 256)

    >>> # Build a custom stack using exported blocks and upsampler
    >>> from opensr_srgan.model.generators import ResidualBlockNoBN, make_upsampler
    >>> body = nn.Sequential(*[ResidualBlockNoBN(96, 3, res_scale=0.2) for _ in range(8)])
    >>> up = make_upsampler(96, scale=4)

Notes:
    - The alias ``flexible_generator`` is kept for convenience and backward
      compatibility; it refers to the class object ``FlexibleGenerator``.
    - Supported scales are {2, 4, 8}. An invalid ``block_type`` will raise a
      ``ValueError`` with the allowed options.
"""

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
