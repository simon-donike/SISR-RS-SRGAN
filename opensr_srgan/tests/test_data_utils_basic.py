"""Sanity checks for :mod:`opensr_srgan.data.data_utils`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import TensorDataset

from opensr_srgan.data import data_utils


def _make_config(**overrides):
    data_defaults = dict(
        dataset_type="dummy",
        train_batch_size=2,
        val_batch_size=3,
        num_workers=0,
        prefetch_factor=2,
    )
    data_defaults.update(overrides)
    return SimpleNamespace(Data=SimpleNamespace(**data_defaults))


def test_datamodule_from_datasets_minimal_workers():
    """The helper should honour explicit batch sizes and worker counts."""

    cfg = _make_config()
    dataset = TensorDataset(torch.arange(8))

    dm = data_utils.datamodule_from_datasets(cfg, dataset, dataset)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 3
    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0


def test_datamodule_from_datasets_prefetch_when_workers_present():
    """``prefetch_factor`` is only forwarded once workers are enabled."""

    cfg = _make_config(num_workers=2, prefetch_factor=4)
    dataset = TensorDataset(torch.arange(12))

    dm = data_utils.datamodule_from_datasets(cfg, dataset, dataset)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert train_loader.num_workers == 2
    assert val_loader.num_workers == 2
    assert train_loader.prefetch_factor == 4
    assert val_loader.prefetch_factor == 4


def test_select_dataset_unknown_raises():
    """Unsupported dataset identifiers should fail loudly."""

    cfg = SimpleNamespace(Data=SimpleNamespace(dataset_type="does-not-exist"))

    with pytest.raises(NotImplementedError):
        data_utils.select_dataset(cfg)