"""Smoke tests for generator and discriminator model variants."""

import pytest

# Torch is an optional dependency for some test environments.
torch = pytest.importorskip("torch")

from opensr_srgan.model.generators import (
    Generator as SRGANGenerator,
    FlexibleGenerator,
    ConditionalGANGenerator,
)
from opensr_srgan.model.discriminators import (
    Discriminator,
    PatchGANDiscriminator,
)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: SRGANGenerator(
                in_channels=4,
                large_kernel_size=5,
                small_kernel_size=3,
                n_channels=16,
                n_blocks=2,
                scaling_factor=4,
            ),
            id="srresnet",
        ),
        *[
            pytest.param(
                lambda block=block_type: FlexibleGenerator(
                    in_channels=4,
                    n_channels=16,
                    n_blocks=2,
                    small_kernel=3,
                    large_kernel=5,
                    scale=4,
                    block_type=block,
                ),
                id=f"flexible-{block_type}",
            )
            for block_type in ("res", "rcab", "rrdb", "lka")
        ],
        pytest.param(
            lambda: ConditionalGANGenerator(
                in_channels=4,
                n_channels=16,
                n_blocks=2,
                small_kernel=3,
                large_kernel=5,
                scale=4,
                noise_dim=8,
                res_scale=0.2,
            ),
            id="conditional-cgan",
        ),
    ],
)
def test_generator_variants_instantiate(factory):
    """Ensure all supported generator variants can be built and run a forward pass."""

    generator = factory()
    lr = torch.randn(1, 4, 128, 128)
    sr = generator(lr)
    assert isinstance(sr, torch.Tensor)
    assert sr.shape[0] == lr.shape[0]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: Discriminator(in_channels=4, n_blocks=4),
            id="standard",
        ),
        pytest.param(
            lambda: PatchGANDiscriminator(input_nc=4, n_layers=3),
            id="patchgan-default",
        ),
        pytest.param(
            lambda: PatchGANDiscriminator(input_nc=4, n_layers=5),
            id="patchgan-deep",
        ),
    ],
)
def test_discriminator_variants_instantiate(factory):
    """Ensure discriminator variants can be built and process a forward pass."""

    discriminator = factory()
    hr = torch.randn(1, 4, 128, 128)
    scores = discriminator(hr)
    assert isinstance(scores, torch.Tensor)
    assert scores.shape[0] == hr.shape[0]
