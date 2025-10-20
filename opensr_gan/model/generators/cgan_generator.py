"""Conditional GAN generator with stochastic latent modulation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..model_blocks import make_upsampler

__all__ = ["ConditionalGANGenerator"]


class NoiseResBlock(nn.Module):
    """Residual block that modulates intermediate features using a latent code."""

    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        noise_dim: int,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.res_scale = res_scale
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_dim, n_channels),
            nn.SiLU(),
            nn.Linear(n_channels, 2 * n_channels),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        style = self.noise_mlp(noise)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        out = self.conv1(x)
        out = out * (1 + gamma) + beta
        out = self.act(out)
        out = self.conv2(out)
        return x + self.res_scale * out


class ConditionalGANGenerator(nn.Module):
    """Generator that conditions on the LR image while injecting stochastic latent noise.

    The forward pass accepts an LR tensor and an optional latent vector. If no latent
    vector is provided, the module samples one from a unit Gaussian. This allows the
    generator to act as a drop-in replacement for existing deterministic generators
    while still supporting explicit control of the random seed during inference.
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_channels: int = 96,
        n_blocks: int = 16,
        small_kernel: int = 3,
        large_kernel: int = 9,
        scale: int = 4,
        noise_dim: int = 128,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()

        if scale not in {2, 4, 8}:
            raise ValueError("scale must be one of {2, 4, 8}")

        self.noise_dim = noise_dim
        self.scale = scale

        padding_large = large_kernel // 2
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, large_kernel, padding=padding_large),
            nn.PReLU(),
        )

        self.body = nn.ModuleList(
            [NoiseResBlock(n_channels, small_kernel, noise_dim, res_scale) for _ in range(n_blocks)]
        )
        self.body_tail = nn.Conv2d(
            n_channels,
            n_channels,
            small_kernel,
            padding=small_kernel // 2,
        )
        self.upsampler = make_upsampler(n_channels, scale)
        self.tail = nn.Conv2d(n_channels, in_channels, large_kernel, padding=padding_large)

    def sample_noise(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Sample latent noise compatible with the module configuration."""

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.randn(batch_size, self.noise_dim, device=device, dtype=dtype)

    def forward(
        self,
        lr: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generate a super-resolved image from an LR input and latent noise."""

        if noise is None:
            noise = torch.randn(
                lr.size(0),
                self.noise_dim,
                device=lr.device,
                dtype=lr.dtype,
            )

        features = self.head(lr)
        residual = features
        for block in self.body:
            residual = block(residual, noise)
        residual = self.body_tail(residual)
        features = features + residual
        features = self.upsampler(features)
        sr = self.tail(features)

        if return_noise:
            return sr, noise
        return sr
