"""Basic instantiation tests for generator architectures."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402  (import after torch availability check)

from opensr_srgan.model.generators import (  # noqa: E402
    ConditionalGANGenerator,
    FlexibleGenerator,
    Generator,
    SRResNet,
)


@pytest.mark.parametrize(
    "generator_cls, kwargs",
    [
        (SRResNet, {}),
        (Generator, {}),
        (FlexibleGenerator, {}),
        (ConditionalGANGenerator, {}),
    ],
)
def test_generator_can_be_instantiated(generator_cls, kwargs):
    """Ensure generator classes can be constructed with default arguments."""

    instance = generator_cls(**kwargs)
    assert isinstance(instance, nn.Module)
