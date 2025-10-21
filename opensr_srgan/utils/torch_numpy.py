"""Fallback helpers for converting Torch tensors to NumPy arrays."""
from __future__ import annotations

import numpy as np
import torch

_TORCH_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.bfloat16: np.float32,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Return a NumPy view or copy of ``tensor`` regardless of PyTorch build.

    Minimal CPU-only wheels of PyTorch that ship without NumPy bindings raise a
    ``RuntimeError`` when :meth:`torch.Tensor.numpy` or :func:`numpy.array`
    attempt to materialise an array.  The tests in this project rely on NumPy
    arrays for verification, so we provide a graceful fallback that goes via
    Python lists when the direct conversion is unavailable.
    """

    tensor_cpu = tensor.detach().cpu()
    if not tensor_cpu.is_contiguous():
        tensor_cpu = tensor_cpu.contiguous()

    try:
        return tensor_cpu.numpy()
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        dtype = _TORCH_TO_NUMPY_DTYPE.get(tensor_cpu.dtype)
        return np.asarray(tensor_cpu.tolist(), dtype=dtype)
