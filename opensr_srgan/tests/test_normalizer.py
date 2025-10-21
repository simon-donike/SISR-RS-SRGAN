"""Tests for the configurable normalizer helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from opensr_srgan.data.utils.normalizer import Normalizer


class DummyConfig(SimpleNamespace):
    """Namespace helper that allows attribute-style access in tests."""


def test_default_method_is_sen2_stretch():
    cfg = DummyConfig(Data=DummyConfig())
    normalizer = Normalizer(cfg)

    tensor = torch.tensor([-0.5, 0.0, 0.75])
    stretched = normalizer.normalize(tensor)
    recovered = normalizer.denormalize(stretched)

    assert normalizer.method == "sen2_stretch"


def test_alias_normalize_10k_is_supported():
    cfg = DummyConfig(Data=DummyConfig(normalization="normalize_10k"))
    normalizer = Normalizer(cfg)

    tensor = torch.tensor([0.0, 1000.0, 2000.0])
    normalized = normalizer.normalize(tensor)
    recovered = normalizer.denormalize(normalized)

    assert normalizer.method == "normalise_10k"
    assert torch.allclose(recovered, torch.clamp(tensor, 0.0, 10000.0), atol=1e-4)


def test_unknown_method_raises_value_error():
    cfg = DummyConfig(Data=DummyConfig(normalization="unsupported"))

    with pytest.raises(ValueError):
        Normalizer(cfg)
