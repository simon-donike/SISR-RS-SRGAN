# opensr_srgan/tests/test_generator_content_loss.py
import torch
import pytest
from omegaconf import OmegaConf
from pathlib import Path
import types

from opensr_srgan.model.loss.loss import GeneratorContentLoss


@pytest.fixture(scope="session")
def cfg_10m():
    cfg_path = Path("opensr_srgan/configs/config_10m.yaml")
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    return OmegaConf.load(cfg_path)


def test_generator_content_loss_forward(monkeypatch, cfg_10m):
    loss_fn = GeneratorContentLoss(cfg_10m, testing=True)

    B, C, H, W = 1, 13, 32, 32
    sr = torch.rand(B, C, H, W)
    hr = torch.rand(B, C, H, W)

    loss, metrics = loss_fn.return_loss(sr, hr)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert all(isinstance(v, torch.Tensor) for v in metrics.values())

    eval_metrics = loss_fn.return_metrics(sr, hr, prefix="val")
    assert any(k.startswith("val/") for k in eval_metrics.keys())
