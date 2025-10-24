"""PatchGAN discriminator adapted from the pix2pix/CycleGAN reference implementation."""

from __future__ import annotations

import functools
from torch import nn

__all__ = ["PatchGANDiscriminator"]


def get_norm_layer(norm_type: str = "instance"):
    """Return a normalization layer factory.

    Parameters
    ----------
    norm_type: str
        Type of normalization: ``"batch"``, ``"instance"`` or ``"none"``.
    """

    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type == "none":

        def _identity(_channels: int):
            return nn.Identity()

        return _identity
    raise NotImplementedError(f"Normalization layer [{norm_type}] is not supported.")


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator that classifies overlapping patches."""

    def __init__(
        self, input_nc: int, ndf: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):  # type: ignore[override]
        return self.model(input)


class PatchGANDiscriminator(nn.Module):
    """Convenience wrapper exposing the PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        n_layers: int = 3,
        norm_type: str = "instance",
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError("PatchGAN discriminator requires at least one layer.")

        ndf = 64
        norm_layer = get_norm_layer(norm_type)
        self.model = NLayerDiscriminator(
            input_nc, ndf=ndf, n_layers=n_layers, norm_layer=norm_layer
        )

        self.base_channels = ndf
        self.kernel_size = 4
        self.n_layers = n_layers

    def forward(self, input):  # type: ignore[override]
        return self.model(input)
