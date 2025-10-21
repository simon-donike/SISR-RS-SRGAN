import pytest
import numpy as np
import torch

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from opensr_srgan.utils import spectral_helpers as sh


def test_normalise_s2_roundtrip():
    original = torch.linspace(0.0, 0.25, steps=6)
    normalized = sh.normalise_s2(original.clone(), stage="norm")
    recovered = sh.normalise_s2(normalized.clone(), stage="denorm")
    assert torch.allclose(recovered, torch.clamp(original, 0, 0.3), atol=1e-4)
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0


def test_normalise_10k_roundtrip():
    original = torch.tensor([0.0, 500.0, 1000.0, 2500.0, 10000.0])
    normalized = sh.normalise_10k(original.clone(), stage="norm")
    recovered = sh.normalise_10k(normalized.clone(), stage="denorm")
    assert torch.allclose(recovered, torch.clamp(original, 0, 10000), atol=1e-4)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0


def test_sen2_stretch_clamps():
    tensor = torch.tensor([-0.1, 0.1, 0.5, 0.9])
    stretched = sh.sen2_stretch(tensor)
    assert stretched.min().item() >= 0.0
    assert stretched.max().item() <= 1.0
    expected = torch.clamp(tensor * (10 / 3.0), 0.0, 1.0)
    assert torch.allclose(stretched, expected)


def test_minmax_percentile_rescales():
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0])
    scaled = sh.minmax_percentile(tensor, pmin=0, pmax=100)
    assert torch.isclose(scaled.min(), torch.tensor(0.0))
    assert torch.isclose(scaled.max(), torch.tensor(1.0))


def test_histogram_preserves_shape_and_dtype():
    reference = torch.tensor([
        [[0.0, 0.2], [0.4, 0.6]],
        [[0.1, 0.3], [0.5, 0.7]],
    ], dtype=torch.float32)
    target = reference.clone()
    matched = sh.histogram(reference, target)
    assert matched.shape == target.shape
    assert matched.dtype == target.dtype
    assert torch.allclose(matched, target, atol=1e-5)


def test_histogram_batched_channel_mismatch_raises():
    reference = torch.rand(1, 3, 4, 4)
    target = torch.rand(1, 4, 4, 4)
    try:
        sh.histogram(reference, target)
    except AssertionError as exc:
        assert "Channel mismatch" in str(exc)
    else:
        raise AssertionError("Expected an assertion for channel mismatch")


def test_moment_matches_statistics():
    reference = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[10.0, 12.0], [14.0, 16.0]],
            [[20.0, 24.0], [28.0, 32.0]],
        ],
        dtype=torch.float32,
    )

    matched = sh.moment(reference, target)
    for ref_ch, matched_ch in zip(reference, matched):
        assert np.isclose(
            np.mean(np.array(matched_ch)), np.mean(np.array(ref_ch)), atol=1e-5
        )
        assert np.isclose(
            np.std(np.array(matched_ch)), np.std(np.array(ref_ch)), atol=1e-5
        )
