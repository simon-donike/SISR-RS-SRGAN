"""Tests ensuring example configs and prebuilt factories can be instantiated."""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")
OmegaConf = omegaconf.OmegaConf

from opensr_srgan.data import data_utils

# --- FIX: give the module a real name & register it in sys.modules ---
_factory_path = ROOT / "_factory.py"
_factory_spec = importlib.util.spec_from_file_location(
    "opensr_srgan._factory", str(_factory_path)
)
assert _factory_spec and _factory_spec.loader
_factory = importlib.util.module_from_spec(_factory_spec)

# Register before executing so dataclasses & relative imports resolve
sys.modules[_factory_spec.name] = _factory
_factory.__package__ = "opensr_srgan"

_factory_spec.loader.exec_module(_factory)  # type: ignore[assignment]


@lru_cache(maxsize=None)
def _instantiate_training_model(config_path: Path):
    """Instantiate the training model once per unique config path."""

    from opensr_srgan.model.SRGAN import SRGAN_model

    return SRGAN_model(config_file_path=str(config_path))

@pytest.mark.parametrize("config_name", ["config_10m.yaml", "config_20m.yaml"])
def test_example_configs_can_instantiate(config_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the training script by loading configs and instantiating model + datamodule."""

    config_path = ROOT / "configs" / config_name
    config = OmegaConf.load(config_path)

    sentinel = object()

    def fake_select_dataset(cfg):  # type: ignore[override]
        assert cfg is config
        return sentinel

    monkeypatch.setattr(data_utils, "select_dataset", fake_select_dataset)

    model = _instantiate_training_model(config_path)
    assert hasattr(model, "generator")
    assert hasattr(model, "discriminator")

    datamodule = data_utils.select_dataset(config)
    assert datamodule is sentinel


@pytest.mark.parametrize("preset", sorted(_factory._PRESETS))
def test_prebuilt_models_can_instantiate(preset: str, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Ensure factory presets can be instantiated without performing remote downloads."""

    config_sources = {
        "RGB": ROOT / "configs" / "config_10m.yaml",
        "SWIR": ROOT / "configs" / "config_20m.yaml",
    }

    download_map: dict[str, Path] = {}
    for meta in _factory._PRESETS.values():
        key = "RGB" if "RGB" in meta.config_filename.upper() else "SWIR"
        config_source = config_sources[key]

        config_target = tmp_path / meta.config_filename
        config_target.write_text(config_source.read_text())
        download_map[meta.config_filename] = config_target

        checkpoint_target = tmp_path / meta.checkpoint_filename
        torch.save({"state_dict": {}}, checkpoint_target)
        download_map[meta.checkpoint_filename] = checkpoint_target

    def fake_hf_hub_download(*, repo_id: str, filename: str, cache_dir=None):
        assert filename in download_map
        return str(download_map[filename])

    hub_module = types.ModuleType("huggingface_hub")
    hub_module.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    model = _factory.load_inference_model(preset, map_location="cpu")
    from opensr_srgan.model.SRGAN import SRGAN_model

    assert isinstance(model, SRGAN_model)
    assert not model.training
