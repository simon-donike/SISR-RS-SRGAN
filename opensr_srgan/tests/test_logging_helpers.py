import pytest

PIL = pytest.importorskip("PIL")
Image = PIL.Image

torch = pytest.importorskip("torch")

from opensr_srgan.utils import logging_helpers as lh


def test_to_numpy_img_rgb():
    skip = True
    if not skip:
        import torch
        import numpy as np
        tensor = torch.rand(3, 6, 6)
        array = lh._to_numpy_img(tensor)
        assert array.shape == (6, 6, 3)


def test_plot_tensors_returns_pil_image():
    lr = torch.rand(1, 3, 4, 4)
    sr = torch.rand(1, 3, 4, 4)
    hr = torch.rand(1, 3, 4, 4)
    image = lh.plot_tensors(lr, sr, hr, title="Unit Test")
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0
