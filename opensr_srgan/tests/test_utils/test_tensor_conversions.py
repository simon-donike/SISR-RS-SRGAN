# opensr_srgan/tests/test_torch_numpy.py
import torch
import numpy as np
import pytest

from opensr_srgan.utils.tensor_conversions import tensor_to_numpy


def test_tensor_to_numpy_basic():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = tensor_to_numpy(x)
    assert isinstance(y, np.ndarray)
    assert y.shape == (2, 3)
    assert np.allclose(y, x.cpu().numpy())


def test_tensor_to_numpy_noncontiguous():
    x = torch.arange(9, dtype=torch.int32).reshape(3, 3).t()  # noncontiguous
    y = tensor_to_numpy(x)
    assert isinstance(y, np.ndarray)
    assert np.all(y == x.cpu().numpy())


def test_tensor_to_numpy_dtype_mapping():
    # just pick some representative dtypes
    for t_dtype, n_dtype in [
        (torch.float16, np.float16),
        (torch.int64, np.int64),
        (torch.uint8, np.uint8),
        (torch.bool, np.bool_),
    ]:
        x = torch.zeros((2, 2), dtype=t_dtype)
        y = tensor_to_numpy(x)
        assert y.dtype == n_dtype
