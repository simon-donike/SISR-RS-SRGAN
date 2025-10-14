"""Exponential Moving Average (EMA) utilities for model parameter smoothing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintain exponential moving averages of a model's parameters.

    The implementation tracks trainable parameters as well as buffers (e.g.,
    running statistics in batch-normalization layers) and provides helpers to
    temporarily swap a model to the smoothed weights during evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        *,
        use_num_updates: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1 (inclusive)")

        self.decay = float(decay)
        self.num_updates = 0 if use_num_updates else None
        self.device = torch.device(device) if device is not None else None

        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}
        self.collected_params: Dict[str, torch.Tensor] = {}
        self.collected_buffers: Dict[str, torch.Tensor] = {}

        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow = param.detach().clone()
            if self.device is not None:
                shadow = shadow.to(self.device)
            self.shadow_params[name] = shadow

        for name, buffer in model.named_buffers():
            shadow = buffer.detach().clone()
            if self.device is not None:
                shadow = shadow.to(self.device)
            self.shadow_buffers[name] = shadow

    def update(self, model: nn.Module) -> None:
        """Update EMA weights using parameters from ``model``."""

        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if name not in self.shadow_params:
                    # Parameter may have been added dynamically (rare, but safeguard)
                    shadow = param.detach().clone()
                    if self.device is not None:
                        shadow = shadow.to(self.device)
                    self.shadow_params[name] = shadow

                shadow_param = self.shadow_params[name]
                param_data = param.detach()
                if param_data.device != shadow_param.device:
                    param_data = param_data.to(shadow_param.device)

                shadow_param.lerp_(param_data, one_minus_decay)

            for name, buffer in model.named_buffers():
                if name not in self.shadow_buffers:
                    shadow = buffer.detach().clone()
                    if self.device is not None:
                        shadow = shadow.to(self.device)
                    self.shadow_buffers[name] = shadow

                shadow_buffer = self.shadow_buffers[name]
                buffer_data = buffer.detach()
                if buffer_data.device != shadow_buffer.device:
                    buffer_data = buffer_data.to(shadow_buffer.device)
                shadow_buffer.copy_(buffer_data)

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into ``model`` while backing up original values."""

        if self.collected_params or self.collected_buffers:
            raise RuntimeError("EMA weights already applied; call restore() before reapplying.")

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow_params:
                continue
            self.collected_params[name] = param.detach().clone()
            param.data.copy_(self.shadow_params[name].to(param.device))

        for name, buffer in model.named_buffers():
            if name not in self.shadow_buffers:
                continue
            self.collected_buffers[name] = buffer.detach().clone()
            buffer.data.copy_(self.shadow_buffers[name].to(buffer.device))

    def restore(self, model: nn.Module) -> None:
        """Restore original parameters that were swapped out via :meth:`apply_to`."""

        for name, param in model.named_parameters():
            cached = self.collected_params.pop(name, None)
            if cached is None:
                continue
            param.data.copy_(cached.to(param.device))

        for name, buffer in model.named_buffers():
            cached = self.collected_buffers.pop(name, None)
            if cached is None:
                continue
            buffer.data.copy_(cached.to(buffer.device))

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Context manager that temporarily applies EMA weights to ``model``."""

        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def to(self, device: str | torch.device) -> None:
        """Move EMA statistics to ``device``."""

        target_device = torch.device(device)
        for name, tensor in list(self.shadow_params.items()):
            self.shadow_params[name] = tensor.to(target_device)
        for name, tensor in list(self.shadow_buffers.items()):
            self.shadow_buffers[name] = tensor.to(target_device)
        self.device = target_device

    def state_dict(self) -> Dict[str, object]:
        """Return a serializable state dict for checkpointing."""

        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "device": str(self.device) if self.device is not None else None,
            "shadow_params": {k: v.detach().cpu() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.detach().cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        """Load EMA statistics from ``state_dict``."""

        self.decay = float(state_dict["decay"])
        self.num_updates = state_dict["num_updates"]
        device_str = state_dict.get("device", None)
        self.device = torch.device(device_str) if device_str is not None else None

        self.shadow_params = {
            name: tensor.clone().to(self.device) if self.device is not None else tensor.clone()
            for name, tensor in state_dict.get("shadow_params", {}).items()
        }
        self.shadow_buffers = {
            name: tensor.clone().to(self.device) if self.device is not None else tensor.clone()
            for name, tensor in state_dict.get("shadow_buffers", {}).items()
        }

        self.collected_params = {}
        self.collected_buffers = {}
