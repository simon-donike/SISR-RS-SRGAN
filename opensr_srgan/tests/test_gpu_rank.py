"""Tests for GPU rank helper utilities."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

pytest.importorskip("numpy")

from opensr_srgan.utils import gpu_rank


def test_is_global_zero_defaults_to_true(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert gpu_rank._is_global_zero() is True


def test_is_global_zero_respects_environment(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert gpu_rank._is_global_zero() is True

    monkeypatch.setenv("RANK", "2")
    assert gpu_rank._is_global_zero() is False


def test_is_global_zero_honours_distributed_module(monkeypatch):
    dummy_dist = ModuleType("torch.distributed")
    dummy_dist.is_available = lambda: True
    dummy_dist.is_initialized = lambda: True
    dummy_dist.get_rank = lambda: 1

    monkeypatch.setitem(sys.modules, "torch.distributed", dummy_dist)

    assert gpu_rank._is_global_zero() is False
