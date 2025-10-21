"""Lightweight smoke tests for :mod:`opensr_srgan.inference`."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from opensr_srgan import inference


class DummyModel:
    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path
        self.state_dict = None
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state, strict=False):
        self.state_dict = (state, strict)

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        instance = cls(config_file_path="from_checkpoint")
        instance.ckpt_args = (args, kwargs)
        return instance


def test_load_model_prefers_available_device(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "SRGAN_model", DummyModel)
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    model, device = inference.load_model(config_path="cfg.yaml")

    assert isinstance(model, DummyModel)
    assert model.config_file_path == "cfg.yaml"
    assert device == "cpu"
    assert model.device == "cpu"


def test_load_model_falls_back_to_raw_state_dict(monkeypatch, tmp_path):
    class RaisingDummy(DummyModel):
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            raise TypeError("unexpected argument")

    ckpt_path = tmp_path / "weights.ckpt"
    ckpt_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(inference, "SRGAN_model", RaisingDummy)

    recorded = {}

    def fake_load(path, map_location=None):
        recorded["path"] = Path(path)
        recorded["map_location"] = map_location
        return {"state_dict": {"weight": 1}}

    monkeypatch.setattr(inference.torch, "load", fake_load)
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: False)

    model, device = inference.load_model(config_path="cfg.yaml", ckpt_path=str(ckpt_path))

    assert recorded["path"] == ckpt_path
    assert recorded["map_location"] == "cpu"
    assert model.state_dict == ({"weight": 1}, False)
    assert device == "cpu"
