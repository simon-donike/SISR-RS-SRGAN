"""Tests for the generator content loss helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from opensr_srgan.model.loss import GeneratorContentLoss


class DummyConfig(SimpleNamespace):
    """Namespace helper mirroring the OmegaConf structure used in training."""


def _build_dummy_cfg() -> DummyConfig:
    """Construct a minimal configuration expected by ``GeneratorContentLoss``."""

    losses = DummyConfig(
        l1_weight=1.0,
        sam_weight=0.1,
        perceptual_weight=0.25,
        tv_weight=0.2,
        max_val=1.0,
        ssim_win=7,
        randomize_bands=False,
        fixed_idx=(0, 1, 2),
        perceptual_metric="vgg",
    )

    training = DummyConfig(Losses=losses)
    truncated_vgg = DummyConfig(i=1, j=1)

    return DummyConfig(Data=DummyConfig(normalization="sen2_stretch"), Training=training, TruncatedVGG=truncated_vgg)


def test_generator_content_loss_cpu(monkeypatch):
    """Instantiate the loss on CPU and evaluate both loss and metric helpers."""

    class _StubVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1))

    monkeypatch.setattr(
        "torchvision.models.VGG19_Weights", SimpleNamespace(DEFAULT=None), raising=False
    )
    monkeypatch.setattr(
        "torchvision.models.vgg19", lambda weights=None: _StubVGG(), raising=False
    )

    cfg = _build_dummy_cfg()
    loss_fn = GeneratorContentLoss(cfg)

    sr = torch.rand(2, 4, 8, 8)
    hr = torch.rand(2, 4, 8, 8)

    total_loss, metrics = loss_fn.return_loss(sr, hr)

    assert total_loss.device.type == "cpu"
    assert set(metrics) == {"l1", "sam", "perceptual", "tv", "psnr", "ssim"}
    for value in metrics.values():
        assert value.device.type == "cpu"
        assert not value.requires_grad

    prefixed_metrics = loss_fn.return_metrics(sr, hr, prefix="train")
    expected_keys = {"train/l1", "train/sam", "train/perceptual", "train/tv", "train/psnr", "train/ssim"}
    assert set(prefixed_metrics) == expected_keys
    for value in prefixed_metrics.values():
        assert value.device.type == "cpu"
        assert not value.requires_grad
