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
    """Convert a PyTorch tensor to a NumPy array safely across environments.

    Provides a robust fallback for minimal or CPU-only PyTorch builds that lack
    NumPy bindings (where ``tensor.numpy()`` raises ``RuntimeError: Numpy is not available``).
    Ensures tensors are detached, moved to CPU, and contiguous before conversion.

    Args:
        tensor (torch.Tensor): Input tensor to convert.

    Returns:
        numpy.ndarray: NumPy array representation of the tensor.
        Falls back to ``tensor.tolist()`` conversion if direct bindings are unavailable.

    Raises:
        RuntimeError: Re-raises conversion errors not related to missing NumPy bindings.

    Notes:
        - Keeps dtype consistency between PyTorch and NumPy via a lookup table.
        - Returns a view when possible, or a copy when conversion via list fallback is used.
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
