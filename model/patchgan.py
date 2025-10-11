"""PatchGAN discriminator tuned for 4× remote-sensing super-resolution."""

from __future__ import annotations

import functools
from torch import nn

__all__ = ["PatchGANDiscriminator"]


def get_norm_layer():
    """Return the normalization layer factory used for PatchGAN."""

    # Instance normalization proved effective for multi-spectral remote-sensing imagery
    # as it keeps per-instance contrast statistics without tracking dataset-wide moments.
    return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies overlapping image patches."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
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
        nf_mult = min(2 ** n_layers, 8)
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):  # type: ignore[override]
        return self.model(input)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with sensible defaults for 4× remote-sensing SR."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer()

        # Clamp to a sane range to avoid degenerate receptive fields.
        depth = max(3, min(n_blocks, 6))
        self.model = NLayerDiscriminator(input_nc, ndf, n_layers=depth, norm_layer=norm_layer)

    def forward(self, input):  # type: ignore[override]
        return self.model(input)
