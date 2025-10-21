"""Unit tests for the lightweight factory helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")


if "pytorch_lightning" not in sys.modules:
    dummy_pl = ModuleType("pytorch_lightning")

    class _DummyLightningModule:
        pass

    dummy_pl.LightningModule = _DummyLightningModule
    sys.modules["pytorch_lightning"] = dummy_pl

from opensr_srgan import _factory


def test_maybe_download_accepts_existing_file(tmp_path):
    ckpt = tmp_path / "weights.ckpt"
    ckpt.write_bytes(b"binary")

    with _factory._maybe_download(ckpt) as resolved:
        assert resolved == ckpt


def test_maybe_download_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError):
        with _factory._maybe_download(missing):
            pass


class DummyEMA:
    def __init__(self):
        self.state = None

    def load_state_dict(self, state):
        self.state = state


class DummySRGAN:
    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path
        self.loaded_state = None
        self.eval_called = False
        self.ema = DummyEMA()

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)

    def eval(self):
        self.eval_called = True
        return self


def test_load_from_config_restores_checkpoint(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("key: value")

    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"ckpt")

    monkeypatch.setattr(_factory, "SRGAN_model", DummySRGAN)

    recorded = {}

    def fake_load(path, map_location=None):
        recorded["path"] = Path(path)
        recorded["map_location"] = map_location
        return {"state_dict": {"weight": 3}, "ema_state": {"ema": 1}}

    monkeypatch.setattr(_factory.torch, "load", fake_load)

    model = _factory.load_from_config(config_path, checkpoint_path, map_location="cpu")

    assert isinstance(model, DummySRGAN)
    assert model.config_file_path == str(config_path)
    assert model.eval_called is True
    assert model.loaded_state == ({"weight": 3}, False)
    assert model.ema.state == {"ema": 1}
    assert recorded["path"] == checkpoint_path
    assert recorded["map_location"] == "cpu"


def test_load_from_config_missing_config(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        _factory.load_from_config(missing)


def test_load_inference_model_unknown_preset():
    with pytest.raises(ValueError):
        _factory.load_inference_model("unknown")
