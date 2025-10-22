# test_build_lightning_kwargs.py
from opensr_srgan.utils.build_trainer_kwargs import build_lightning_kwargs

def test_build_lightning_kwargs():
    from omegaconf import OmegaConf
    config = OmegaConf.load("opensr_srgan/configs/config_10m.yaml")
    
    # Mock logger and callbacks
    mock_logger = None
    mock_checkpoint_callback = None
    mock_early_stop_callback = None
    
    build_lightning_kwargs(config,
    logger=mock_logger,
    checkpoint_callback=mock_checkpoint_callback,
    early_stop_callback=mock_early_stop_callback,
    resume_ckpt=None)
    
if __name__ == "__main__":
    test_build_lightning_kwargs()