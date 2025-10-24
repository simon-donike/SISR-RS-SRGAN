# opensr_srgan/tests/test_model_summary.py
import torch
from types import SimpleNamespace
import yaml
from pathlib import Path
from omegaconf import OmegaConf


from opensr_srgan.utils.model_descriptions import print_model_summary


class DummyNet(torch.nn.Module):
    def __init__(self, n_params=10):
        super().__init__()
        self.layer = torch.nn.Linear(n_params, n_params)
        self.scale = 4
        self.n_blocks = 5
        self.n_layers = 5
        self.base_channels = 64
        self.kernel_size = 3
        self.fc_size = 256

    def forward(self, x):
        return x


# ---- Dummy model container ----
class DummyModel:
    def __init__(self, cfg):
        self.config = cfg
        self.generator = DummyNet()
        self.discriminator = DummyNet()
        self.device = "cpu"
        self.pretrain_g_only = False
        self.g_pretrain_steps = 1000
        self.adv_loss_ramp_steps = 2000
        self.adv_target = 0.9
        self.content_loss_criterion = torch.nn.L1Loss()
        self.adversarial_loss_criterion = torch.nn.BCELoss()


def test_print_model_summary(tmp_path):
    # ---- Load example config ----
    config_path = Path("opensr_srgan/configs/config_10m.yaml")
    conf = OmegaConf.load(config_path)
    model = DummyModel(conf)
    # ---- Run the summary ----
    print_model_summary(model)
