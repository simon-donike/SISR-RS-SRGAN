from opensr_srgan.model.SRGAN import SRGAN_model
from omegaconf import OmegaConf


def test_srgans_creation():

    # 10m
    config = OmegaConf.load("opensr_srgan/configs/config_10m.yaml")
    model = SRGAN_model(config)
    assert model is not None
    assert isinstance(model, SRGAN_model)

    # 20m
    config = OmegaConf.load("opensr_srgan/configs/config_20m.yaml")
    model = SRGAN_model(config)
    assert model is not None
    assert isinstance(model, SRGAN_model)
