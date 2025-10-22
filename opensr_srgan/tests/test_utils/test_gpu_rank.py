"""Tests for GPU rank helper utilities."""

# test_is_global_zero.py
import builtins
import types
import pytest
import sys

# Import the function to be tested
from opensr_srgan.utils.gpu_rank import _is_global_zero


def test_returns_true_when_no_torch_and_no_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setitem(builtins.__dict__, "importlib", None)
    assert _is_global_zero() is True


def test_returns_true_when_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _is_global_zero() is True


def test_returns_false_when_not_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _is_global_zero() is False


def test_returns_true_when_world_size_one(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert _is_global_zero() is True
