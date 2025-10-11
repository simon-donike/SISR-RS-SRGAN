# General Imports
import torch
torch.set_float32_matmul_precision('medium')
import multiprocessing
from multiprocessing import freeze_support

import os
from omegaconf import OmegaConf
import datetime

# PL imports
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

# Local Imports
from utils.datasets import dataset_selector
from model.fusion import RecursiveNet_pl


if __name__ == '__main__':
    # required for Multprocessing on Windows
    freeze_support()


    # load config file
    config = OmegaConf.load("config.yaml")

    # get dataset
    pl_datamodule = dataset_selector(config)

    # get model
    fusion_model = RecursiveNet_pl()

    # Loggers
    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.normpath('logs/tmp'))
    from pytorch_lightning.loggers import WandbLogger
    wandb_project = "2023_SRGAN" # "testing"
    wandb_logger = WandbLogger(project=wandb_project,entity="simon-donike")

    # perform some training
    dir_save_checkpoints = os.path.join(tb_logger.save_dir,wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Experiment Path:",dir_save_checkpoints)
    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,    
                                            monitor='val/L2',
                                            mode='min',
                                            save_last=True,
                                            save_top_k=2)


    # define trainer
    trainer = Trainer(accelerator='cuda',
                    devices=[0],
                    check_val_every_n_epoch=1,
                    val_check_interval=0.5,
                    limit_val_batches=5,
                    max_epochs=100,
                    logger=[wandb_logger],
                    callbacks=[ checkpoint_callback])
    # fit model
    trainer.fit(fusion_model, datamodule=pl_datamodule)


