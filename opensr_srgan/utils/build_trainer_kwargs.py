# put this near the top of your module (e.g., in opensr_srgan/train.py)
import os
import inspect
import torch
import pytorch_lightning as pl
from packaging.version import Version

def build_lightning_kwargs(
    config,
    logger,
    checkpoint_callback,
    early_stop_callback,
    resume_ckpt: str | None = None,
):
    """Return (trainer_kwargs, fit_kwargs) compatible with PL <2 and >=2."""
    # 1) Version + env cleanup
    is_v2 = Version(pl.__version__) >= Version("2.0.0")
    # Prevent legacy env from injecting removed arg on PL>=2
    os.environ.pop("PL_TRAINER_RESUME_FROM_CHECKPOINT", None)

    # 2) Devices / strategy
    devices_cfg = config.Training.gpus  # can be int or list
    if isinstance(devices_cfg, int):
        ndev = devices_cfg
    elif isinstance(devices_cfg, (list, tuple)):
        ndev = len(devices_cfg)
    else:
        ndev = 1
    strategy = "ddp" if ndev > 1 else None
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # 3) Build base kwargs
    trainer_kwargs = dict(
        accelerator=accelerator,
        strategy=strategy,  # may be None -> we’ll drop it
        devices=devices_cfg,
        val_check_interval=config.Training.val_check_interval,
        limit_val_batches=config.Training.limit_val_batches,
        max_epochs=config.Training.max_epochs,
        log_every_n_steps=100,
        logger=[logger],
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    # 4) Drop None-valued keys (fixes strategy=None)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    # 5) Add legacy resume kwarg only on PL<2
    if not is_v2 and resume_ckpt:
        trainer_kwargs["resume_from_checkpoint"] = resume_ckpt

    # 6) Filter by current Trainer.__init__ signature (extra safety)
    init_sig = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in init_sig}

    # 7) Build fit kwargs for PL>=2
    fit_kwargs = {}
    if is_v2 and resume_ckpt:
        fit_sig = inspect.signature(pl.Trainer.fit).parameters
        if "ckpt_path" in fit_sig:
            fit_kwargs["ckpt_path"] = resume_ckpt

    return trainer_kwargs, fit_kwargs
