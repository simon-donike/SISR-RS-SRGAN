"""Discriminator architectures for SRGAN."""

from .srgan_discriminator import Discriminator
from .patchgan import PatchGANDiscriminator

__all__ = [
    "Discriminator",
    "PatchGANDiscriminator",
]
