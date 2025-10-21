"""Utility helpers for configuring tensor normalization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...utils.spectral_helpers import normalise_10k, sen2_stretch


@dataclass
class _NormalizerConfig:
    """Lightweight wrapper around config lookups.

    Parameters
    ----------
    method: str
        Name of the normalization strategy requested via the user config.
    """

    method: str


class Normalizer:
    """Factory for applying configurable normalization/denormalization.

    The normalizer inspects the provided configuration, determines the
    requested normalization scheme, and exposes ``normalize`` / ``denormalize``
    helpers that downstream components can reuse without importing
    :mod:`utils.spectral_helpers` directly.

    Supported methods include Sentinel-2 specific helpers such as
    ``"sen2_stretch"`` (contrast stretch) and ``"normalise_10k"`` (scale
    reflectance-like values stored in the ``[0, 10000]`` range).
    """

    _SUPPORTED_METHODS = {"normalise_10k", "sen2_stretch"}

    def __init__(self, config: Any):
        data_cfg = getattr(config, "Data", None)
        method = None
        if data_cfg is not None:
            method = getattr(data_cfg, "normalization", None)
            if method is None and isinstance(data_cfg, dict):
                method = data_cfg.get("normalization")
        if method is None:
            method = "sen2_stretch"

        method = str(method).strip().lower()
        if method == "normalize_10k":
            method = "normalise_10k"
        if method not in self._SUPPORTED_METHODS:
            supported = ", ".join(sorted(self._SUPPORTED_METHODS))
            raise ValueError(
                f"Unsupported normalization '{method}'. Supported methods: {supported}."
            )

        self._cfg = _NormalizerConfig(method=method)

    @property
    def method(self) -> str:
        """Return the normalization method configured for this instance."""

        return self._cfg.method

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize ``tensor`` according to the configured method."""

        if self.method == "sen2_stretch":
            return sen2_stretch(tensor)
        if self.method == "normalise_10k":
            return normalise_10k(tensor, stage="norm")
        raise RuntimeError(f"Unhandled normalization method: {self.method}")

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert the normalization previously applied by :meth:`normalize`."""

        if self.method == "sen2_stretch":
            return torch.clamp(tensor * (3.0 / 10.0), 0.0, 1.0)
        if self.method == "normalise_10k":
            return normalise_10k(tensor, stage="denorm")
        raise RuntimeError(f"Unhandled normalization method: {self.method}")
