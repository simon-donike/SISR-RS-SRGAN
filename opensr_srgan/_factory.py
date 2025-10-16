"""Utility helpers to instantiate pretrained SRGAN models."""

from __future__ import annotations

import dataclasses
import tempfile
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional, Union

import torch
from pytorch_lightning import LightningModule

from model.SRGAN import SRGAN_model

__all__ = ["instantiate_from_config", "load_inference_model"]


@dataclasses.dataclass(frozen=True)
class _Preset:
    """Metadata describing an inference-ready configuration."""

    config_resource: resources.abc.Traversable
    repo_id: str
    filename: str


_PRESETS = {
    "RGB-NIR": _Preset(
        config_resource=resources.files("opensr_srgan.configs").joinpath("rgb_nir.yaml"),
        repo_id="ESAOpenSR/opensr-srgan-rgb-nir",
        filename="srgan_rgb-nir.ckpt",
    ),
    "SWIR": _Preset(
        config_resource=resources.files("opensr_srgan.configs").joinpath("swir.yaml"),
        repo_id="ESAOpenSR/opensr-srgan-swir",
        filename="srgan_swir.ckpt",
    ),
}


@contextmanager
def _maybe_download(checkpoint_uri: Union[str, Path]) -> Iterator[Path]:
    """Resolve a checkpoint URI to a local file path."""

    uri_str = str(checkpoint_uri)
    candidate = Path(checkpoint_uri)
    if candidate.is_file():
        yield candidate
        return

    if uri_str.startswith(("http://", "https://")):
        suffix = candidate.suffix if candidate.suffix else ".ckpt"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            torch.hub.download_url_to_file(uri_str, tmp.name, progress=False)
            tmp.flush()
            yield Path(tmp.name)
        return

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_uri}' does not exist. Provide a local path or HTTP(S) URL."
    )


def instantiate_from_config(
    config_path: Union[str, Path],
    checkpoint_uri: Optional[Union[str, Path]] = None,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> LightningModule:
    """Instantiate ``SRGAN_model`` from a YAML config and optional checkpoint.

    Parameters
    ----------
    config_path:
        Filesystem path to the YAML configuration that describes the generator
        and discriminator architecture. This should match the configuration the
        checkpoint was trained with.
    checkpoint_uri:
        Optional path or HTTP(S) URL pointing to a Lightning checkpoint. When
        omitted, the factory returns an untrained model initialised from the
        supplied config.
    map_location:
        Optional argument forwarded to :func:`torch.load` during checkpoint
        deserialisation.
    """

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file '{config_path}' could not be located.")

    if checkpoint_uri is None:
        model = SRGAN_model(config_file_path=str(config_path))
    else:
        with _maybe_download(checkpoint_uri) as resolved_path:
            model = SRGAN_model.load_from_checkpoint(
                str(resolved_path),
                config_file_path=str(config_path),
                map_location=map_location,
            )

    model.eval()
    return model


def load_inference_model(
    preset: str,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> LightningModule:
    """Instantiate an off-the-shelf pretrained SRGAN.

    The function downloads a known-good configuration + checkpoint pair from
    the Hugging Face Hub (unless it is already cached) and restores the
    packaged Lightning module.
    """

    key = preset.strip().replace("_", "-").upper()
    try:
        preset_meta = _PRESETS[key]
    except KeyError as err:
        valid = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Available options: {valid}.") from err

    try:  # pragma: no cover - import guard only used at runtime
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "huggingface_hub is required for load_inference_model. "
            "Install the project extras or run 'pip install huggingface-hub'."
        ) from exc

    with resources.as_file(preset_meta.config_resource) as config_path:
        checkpoint_path = hf_hub_download(
            repo_id=preset_meta.repo_id,
            filename=preset_meta.filename,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )

        return instantiate_from_config(
            config_path,
            checkpoint_path,
            map_location=map_location,
        )
