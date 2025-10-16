"""Top-level package for OpenSR SRGAN deployment helpers."""

from __future__ import annotations

from model.SRGAN import SRGAN_model

from ._factory import load_model

__all__ = [
    "SRGAN_model",
    "load_model",
]
