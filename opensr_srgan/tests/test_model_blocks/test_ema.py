# opensr_srgan/tests/test_ema.py
import torch
from torch import nn
import pytest

from opensr_srgan.model.model_blocks.EMA import ExponentialMovingAverage


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(1))

    def forward(self, x):
        return self.lin(x) * self.scale


def test_register_and_state_dict():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    # all named parameters registered
    assert set(ema.shadow_params.keys()) == {"lin.weight", "lin.bias"}
    assert set(ema.shadow_buffers.keys()) == {"scale"}
    state = ema.state_dict()
    assert "decay" in state and "shadow_params" in state


def test_update_moves_toward_model():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.5, use_num_updates=False)
    old_shadow = ema.shadow_params["lin.weight"].clone()
    # modify model weights
    with torch.no_grad():
        model.lin.weight.add_(1.0)
    ema.update(model)
    new_shadow = ema.shadow_params["lin.weight"]
    # shadow should have increased but not fully matched
    diff = (new_shadow - old_shadow).abs().mean()
    assert diff > 0 and diff < 1.0


def test_apply_and_restore_swap():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    # change shadow so we can detect the swap
    with torch.no_grad():
        ema.shadow_params["lin.weight"].add_(10.0)

    original_weight = model.lin.weight.clone()
    ema.apply_to(model)
    assert torch.allclose(model.lin.weight, ema.shadow_params["lin.weight"])
    ema.restore(model)
    assert torch.allclose(model.lin.weight, original_weight)


def test_average_parameters_context_manager():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    with ema.average_parameters(model):
        # inside context, weights are swapped
        assert torch.allclose(model.lin.weight, ema.shadow_params["lin.weight"])
    # restored after exit
    assert not ema.collected_params


def test_to_device_and_load_state_dict():
    model = TinyNet()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.to("cpu")  # no-op but covers code path

    state = ema.state_dict()
    ema2 = ExponentialMovingAverage(model, decay=0.1)
    ema2.load_state_dict(state)

    for k in ema.shadow_params:
        assert torch.allclose(ema.shadow_params[k], ema2.shadow_params[k])
    assert ema2.decay == pytest.approx(ema.decay)
    assert ema2.num_updates == ema.num_updates


def test_invalid_decay_raises():
    model = TinyNet()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(model, decay=1.5)
