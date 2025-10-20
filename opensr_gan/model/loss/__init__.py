"""Loss components and utilities for SRGAN models."""

from .loss import GeneratorContentLoss
from .vgg import TruncatedVGG19

__all__ = [
    "GeneratorContentLoss",
    "TruncatedVGG19",
]
