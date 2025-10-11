# Package Imports
import torch
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import wandb
import os, datetime
from multiprocessing import freeze_support

# set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# local imports
from model.SRGAN import SRGAN_model

# Run Main Function
if __name__ == '__main__':
    # required for Multprocessing on Windows
    freeze_support()

    # General
    torch.set_float32_matmul_precision('medium')
    # load config
    cfg_filepath = "configs/config.yaml"
    config = OmegaConf.load(cfg_filepath)

    #############################################################################################################
    " LOAD MODEL "
    #############################################################################################################
    # load pretrained or instanciate new
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
    from data.data_utils import select_dataset
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

    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.normpath("logs/"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.normpath('logs/tmp'))

    from pytorch_lightning.callbacks import ModelCheckpoint
    dir_save_checkpoints = os.path.join(tb_logger.save_dir,wandb_project,
                                                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Experiment Path:",dir_save_checkpoints)

    checkpoint_callback = ModelCheckpoint(dirpath=dir_save_checkpoints,
                                            monitor=config.Schedulers.metric,
                                        mode='min',
                                        save_last=True,
                                        save_top_k=2)

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # callback to set up early stopping
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(monitor=config.Schedulers.metric, min_delta=0.00, patience=250, verbose=True,
                                    mode="min",check_finite=True) # patience in epochs

    #############################################################################################################
    """ Start Training """
    #############################################################################################################
    
    trainer = Trainer(accelerator='cuda',
                    devices=[0],
                    check_val_every_n_epoch=1,
                    val_check_interval=1.0,
                    limit_val_batches=250,
                    resume_from_checkpoint=resume_from_checkpoint,
                    max_epochs=99999,
                    logger=[ 
                                wandb_logger,
                            ],
                    callbacks=[ checkpoint_callback,
                                early_stop_callback,
                                lr_monitor
                                ],)


    trainer.fit(model, datamodule=pl_datamodule)
    wandb.finish()
    writer.close()


