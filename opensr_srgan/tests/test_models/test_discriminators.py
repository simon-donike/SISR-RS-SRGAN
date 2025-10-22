"""Basic instantiation tests for discriminator architectures."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from opensr_srgan.model.discriminators import Discriminator, PatchGANDiscriminator  # noqa: E402


@pytest.mark.parametrize(
    "discriminator_cls, kwargs",
    [
        (Discriminator, {}),
        (PatchGANDiscriminator, {"input_nc": 3}),
    ],
)
def test_discriminator_can_be_instantiated(discriminator_cls, kwargs):
    """Ensure discriminator classes can be constructed with the provided arguments."""

    instance = discriminator_cls(**kwargs)
    assert isinstance(instance, nn.Module)
