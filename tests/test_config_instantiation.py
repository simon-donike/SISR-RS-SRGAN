"""Tests ensuring example configs and prebuilt factories can be instantiated."""

from __future__ import annotations

import sys
import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")
OmegaConf = omegaconf.OmegaConf

from opensr_srgan.data import data_utils

# --- FIX: give the module a real name & register it in sys.modules ---
_factory_path = ROOT / "opensr_gan" / "_factory.py"
_factory_spec = importlib.util.spec_from_file_location(
    "opensr_gan._factory", str(_factory_path)  # <<< changed
)
assert _factory_spec and _factory_spec.loader
_factory = importlib.util.module_from_spec(_factory_spec)

# Register before executing so dataclasses & relative imports resolve
sys.modules[_factory_spec.name] = _factory               # <<< changed
_factory.__package__ = "opensr_gan"                    # <<< changed

_factory_spec.loader.exec_module(_factory)  # type: ignore[assignment]

@pytest.mark.parametrize("config_name", ["config_10m.yaml", "config_20m.yaml"])
def test_example_configs_can_instantiate(config_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the training script by loading configs and instantiating model + datamodule."""

    config_path = Path("opensr_gan") / "configs" / config_name
    config = OmegaConf.load(config_path)

    sentinel = object()

    def fake_select_dataset(cfg):  # type: ignore[override]
        assert cfg is config
        return sentinel

    monkeypatch.setattr(data_utils, "select_dataset", fake_select_dataset)

    from opensr_srgan.model.SRGAN import SRGAN_model

    model = SRGAN_model(config_file_path=str(config_path))
    assert hasattr(model, "generator")
    assert hasattr(model, "discriminator")

    datamodule = data_utils.select_dataset(config)
    assert datamodule is sentinel


@pytest.mark.parametrize("preset", sorted(_factory._PRESETS))
def test_prebuilt_models_can_instantiate(preset: str, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Ensure factory presets can be instantiated without performing remote downloads."""

    config_sources = {
        "RGB": Path("opensr_gan/configs/config_10m.yaml"),
        "SWIR": Path("opensr_gan/configs/config_20m.yaml"),
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

    from opensr_srgan.model.SRGAN import SRGAN_model

    model = _factory.load_inference_model(preset, map_location="cpu")
    assert isinstance(model, SRGAN_model)
    assert not model.training
