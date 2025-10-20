"""Generator architectures for SRGAN."""

from .srresnet import SRResNet, Generator
from .flexible_generator import FlexibleGenerator, flexible_generator
from .cgan_generator import ConditionalGANGenerator
from .SRGAN_advanced import *  # re-export compatibility symbols

__all__ = [
    "SRResNet",
    "Generator",
    "FlexibleGenerator",
    "flexible_generator",
    "ConditionalGANGenerator",
]
