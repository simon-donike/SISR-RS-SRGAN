"""OpenSR-GAN package."""

from .model.SRGAN import SRGAN_model
from ._factory import load_from_config, load_inference_model

__all__ = [
    "SRGAN_model",
    "load_from_config",
    "load_inference_model",
]
