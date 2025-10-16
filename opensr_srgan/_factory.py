"""Utility helpers to instantiate pretrained SRGAN models."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

import torch
from pytorch_lightning import LightningModule

from model.SRGAN import SRGAN_model

__all__ = ["load_model"]


@contextmanager
def _maybe_download(checkpoint_uri: Union[str, Path]) -> Iterator[Path]:
    """Resolve a checkpoint URI to a local file path.

    The helper accepts local filesystem paths as well as HTTP(S) URLs. Remote
    checkpoints are downloaded to a temporary file that is cleaned up
    automatically once the caller exits the context manager.
    """

    checkpoint_uri = Path(checkpoint_uri)
    if checkpoint_uri.is_file():
        yield checkpoint_uri
        return

    uri_str = str(checkpoint_uri)
    if uri_str.startswith(("http://", "https://")):
        with tempfile.NamedTemporaryFile(suffix=checkpoint_uri.suffix or ".ckpt") as tmp:
            torch.hub.download_url_to_file(uri_str, tmp.name, progress=False)
            tmp.flush()
            yield Path(tmp.name)
        return

    raise FileNotFoundError(
        f"Checkpoint '{checkpoint_uri}' does not exist. Provide a local path or HTTP(S) URL."
    )


def load_model(
    config_path: Union[str, Path],
    checkpoint_uri: Union[str, Path],
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> LightningModule:
    """Instantiate ``SRGAN_model`` and load pretrained weights.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file that defines the generator and
        discriminator architecture. The same config that was used during
        training must be supplied here.
    checkpoint_uri:
        Location of the Lightning checkpoint to restore. Both filesystem paths
        and HTTP(S) URLs are supported.
    map_location:
        Optional device string or :class:`torch.device` passed to
        :func:`torch.load` while materialising the checkpoint. Use ``"cpu"`` to
        force CPU inference when CUDA is unavailable.

    Returns
    -------
    pytorch_lightning.LightningModule
        A fully constructed :class:`~model.SRGAN.SRGAN_model` with weights
        loaded from ``checkpoint_uri`` and switched to evaluation mode.
    """

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file '{config_path}' could not be located.")

    with _maybe_download(checkpoint_uri) as resolved_path:
        model = SRGAN_model.load_from_checkpoint(
            str(resolved_path),
            config_file_path=str(config_path),
            map_location=map_location,
        )

    model.eval()
    return model
