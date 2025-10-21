# Package Imports
import datetime
import os
from pathlib import Path
from multiprocessing import freeze_support

import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

# set visible GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    import argparse
    from multiprocessing import freeze_support

    # required for Multiprocessing on Windows
    freeze_support()

    # ---- CLI: single positional argument (config path) ----
    parser = argparse.ArgumentParser(description="Train SRGAN with a YAML config.")
    try:
        default_config = Path(__file__).resolve().parent / "configs" / "config_10m.yaml"
    except NameError:
        default_config = Path.cwd() / "opensr_srgan" / "configs" / "config_10m.yaml"
    parser.add_argument(
        "--config", "-c",
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config})"
    )
    args = parser.parse_args()

    # General
    torch.set_float32_matmul_precision('medium')
    # load config
    cfg_filepath = args.config
    config = OmegaConf.load(cfg_filepath)
    
    # Get devices
    cuda_devices = config.Training.gpus
    cuda_strategy = 'ddp' if len(cuda_devices) > 1 else None

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################
    # load pretrained or instanciate new
    from opensr_srgan.model.SRGAN import SRGAN_model
    if config.Model.load_checkpoint != False:
        model = SRGAN_model.load_from_checkpoint(config.Model.load_checkpoint, strict=False)
    else:
        model = SRGAN_model(config_file_path=cfg_filepath)
    if config.Model.continue_training != False:
        resume_from_checkpoint = config.Model.continue_training
    else:
        resume_from_checkpoint = None

    #############################################################################################################
    """ GET DATA """
    #############################################################################################################
    # create dataloaders via dataset_selector -> config -> class selection -> convert to pl_module
    from opensr_srgan.data.data_utils import select_dataset
    pl_datamodule = select_dataset(config)

    #############################################################################################################
    """ Configure Trainer """
    #############################################################################################################
    # set up logging
    from pytorch_lightning.loggers import WandbLogger
    wandb_project = "SRGAN_6bands"   # whatever you want
    wandb_logger = WandbLogger(
        project=wandb_project,
        entity="opensr", 
        log_model=False
    )

    from pytorch_lightning.callbacks import ModelCheckpoint
    dir_save_checkpoints = os.path.join(os.path.normpath("logs/"),wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    from opensr_srgan.utils.gpu_rank import _is_global_zero # make dir only on main process
    if _is_global_zero(): # only on main process
        os.makedirs(dir_save_checkpoints, exist_ok=True)
        print("Experiment Path:",dir_save_checkpoints)
        with open(os.path.join(dir_save_checkpoints, "config.yaml"), 'w') as f: # save config to experiment folder
            OmegaConf.save(config, f)

    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                            monitor=config.Schedulers.metric,
                                        mode='min',
                                        save_last=True,
                                        save_top_k=2)

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(monitor=config.Schedulers.metric, min_delta=0.00, patience=250, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs

    #############################################################################################################
    """ Start Training """
    #############################################################################################################
    
    trainer = Trainer(accelerator='cuda',
                    strategy=cuda_strategy,
                    devices=cuda_devices,
                    val_check_interval=config.Training.val_check_interval,
                    limit_val_batches=config.Training.limit_val_batches,
                    resume_from_checkpoint=resume_from_checkpoint,
                    max_epochs=config.Training.max_epochs,
                    log_every_n_steps=100, # log batch frequency
                    logger=[ 
                                wandb_logger,
                            ],
                    callbacks=[ checkpoint_callback,
                                early_stop_callback,
                                ],)


    trainer.fit(model, datamodule=pl_datamodule)
    wandb.finish()


