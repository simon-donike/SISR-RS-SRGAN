import pytest

PIL = pytest.importorskip("PIL")
Image = PIL.Image

torch = pytest.importorskip("torch")

from opensr_srgan.utils import logging_helpers as lh


def test_to_numpy_img_single_channel():
    tensor = torch.linspace(0, 1, steps=4).view(1, 2, 2)
    array = lh._to_numpy_img(tensor)
    assert array.shape == (2, 2)
    assert array.min() >= 0.0 and array.max() <= 1.0


def test_to_numpy_img_rgb():
    tensor = torch.stack([torch.zeros(2, 2), torch.ones(2, 2), torch.full((2, 2), 0.5)])
    array = lh._to_numpy_img(tensor)
    assert array.shape == (2, 2, 3)
    assert array[..., 0].min() == 0.0
    assert array[..., 1].max() == 1.0


def test_to_numpy_img_invalid_dim():
    tensor = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        lh._to_numpy_img(tensor)


def test_plot_tensors_returns_pil_image():
    lr = torch.rand(1, 3, 4, 4)
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    image = lh.plot_tensors(lr, sr, hr, title="Unit Test")
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0
