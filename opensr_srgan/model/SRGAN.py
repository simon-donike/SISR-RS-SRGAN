# Package Imports
import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType


import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local imports
from opensr_srgan.utils.logging_helpers import plot_tensors
from opensr_srgan.utils.model_descriptions import print_model_summary
from opensr_srgan.utils.radiometrics import histogram as histogram_match
from opensr_srgan.utils.radiometrics import normalise_10k
from opensr_srgan.model.model_blocks import ExponentialMovingAverage


#############################################################################################################
# Basic SRGAN Model with flexible Generator/Discriminator, scalable losses, pretraining, and ramp-up
#############################################################################################################
class SRGAN_model(pl.LightningModule):
    """
    SRGAN_model — PyTorch Lightning implementation of a Super-Resolution GAN.
    
    This class defines a complete SRGAN training setup, including:
      - Generator and Discriminator model initialization
      - Optional VGG-based perceptual (content) loss network
      - Adversarial loss configuration and label smoothing
      - Support for pretraining and progressive adversarial loss ramp-up
      - Integration with a YAML-based configuration system
    
    Args:
        config_file_path (str): Path to the YAML configuration file.
    """

    def __init__(self, config="config.yaml", mode="train"):
        super(SRGAN_model, self).__init__()

        # ======================================================================
        # SECTION: Load Configuration
        # Purpose: Load and parse model/training hyperparameters from YAML file.
        # ======================================================================
        if isinstance(config, str) or isinstance(config, Path):
            config = OmegaConf.load(config)     
        elif isinstance(config, dict):
            config = OmegaConf.create(config)
        elif OmegaConf.is_config(config):
            pass
        else:
            raise TypeError("Config must be a filepath (str or Path), dict, or OmegaConf object.")
        assert mode in {"train", "eval"}, "Mode must be 'train' or 'eval'"  # validate mode
        
        
        # ======================================================================
        # SECTION: Set Variables
        # Purpose: Set config and mode variables model-wide, including PL version.
        # ======================================================================    
        self.config = config
        self.mode = mode          
        self.pl_version = tuple(int(x) for x in pl.__version__.split("."))

        # ======================================================================
        # SECTION: Get Training settings
        # Purpose: Define model variables to enable training strategies.
        # ======================================================================        
        self.pretrain_g_only = bool(getattr(self.config.Training, "pretrain_g_only", False))  # pretrain generator only (default False)
        self.g_pretrain_steps = int(getattr(self.config.Training, "g_pretrain_steps", 0))     # number of steps for G pretraining
        self.adv_loss_ramp_steps = int(getattr(self.config.Training, "adv_loss_ramp_steps", 20000))  # linear ramp-up steps for adversarial loss
        self.adv_target = 0.9 if getattr(self.config.Training, "label_smoothing", False) else 1.0    # use 0.9 if label smoothing enabled, else 1.0
        
        # ======================================================================
        # SECTION: Set up Training Strategy
        # Purpose: Depending on PL version, set up optimizers, schedulers, etc.
        # ======================================================================
        self.setup_lightning()  # dynamically builds and attaches generator + discriminator

        # ======================================================================
        # SECTION: Initialize Generator
        # Purpose: Build generator network depending on selected architecture.
        # ======================================================================
        self.get_models(mode=self.mode)  # dynamically builds and attaches generator + discriminator

        # ======================================================================
        # SECTION: Initialize EMA
        # Purpose: Optional exponential moving average (EMA) tracking for generator weights
        # ======================================================================
        self.initialize_ema()

        # ======================================================================
        # SECTION: Define Loss Functions
        # Purpose: Configure generator content loss and discriminator adversarial loss.
        # ======================================================================
        if self.mode == "train":
            from opensr_srgan.model.loss import GeneratorContentLoss
            self.content_loss_criterion = GeneratorContentLoss(self.config)  # perceptual loss (VGG + pixel)
            self.adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss()   # binary cross-entropy for D/G

    def get_models(self, mode):
        """
        Initialize and attach Generator and Discriminator models based on config.
        Supports multiple generator architectures (SRResNet, RCAB, RRDB, etc.).
        """

        # ======================================================================
        # SECTION: Initialize Generator
        # Purpose: Build generator network depending on selected architecture.
        # ======================================================================
        generator_type = self.config.Generator.model_type

        if generator_type == 'SRResNet':
            # Standard SRResNet generator
            from opensr_srgan.model.generators.srresnet import Generator
            self.generator = Generator(
                in_channels=self.config.Model.in_bands,                # number of input channels
                large_kernel_size=self.config.Generator.large_kernel_size,
                small_kernel_size=self.config.Generator.small_kernel_size,
                n_channels=self.config.Generator.n_channels,
                n_blocks=self.config.Generator.n_blocks,
                scaling_factor=self.config.Generator.scaling_factor
            )
        elif generator_type in ['res', 'rcab', 'rrdb', 'lka']:
            # Advanced generator variants (ResNet, RCAB, RRDB, etc.)
            from opensr_srgan.model.generators.flexible_generator import FlexibleGenerator
            self.generator = FlexibleGenerator(
                in_channels=self.config.Model.in_bands,
                n_channels=self.config.Generator.n_channels,
                n_blocks=self.config.Generator.n_blocks,
                small_kernel=self.config.Generator.small_kernel_size,
                large_kernel=self.config.Generator.large_kernel_size,
                scale=self.config.Generator.scaling_factor,
                block_type=self.config.Generator.model_type
            )
        elif generator_type.lower() in ['conditional_cgan', 'cgan']:
            from opensr_srgan.model.generators import ConditionalGANGenerator

            self.generator = ConditionalGANGenerator(
                in_channels=self.config.Model.in_bands,
                n_channels=self.config.Generator.n_channels,
                n_blocks=self.config.Generator.n_blocks,
                small_kernel=self.config.Generator.small_kernel_size,
                large_kernel=self.config.Generator.large_kernel_size,
                scale=self.config.Generator.scaling_factor,
                noise_dim=getattr(self.config.Generator, "noise_dim", 128),
                res_scale=getattr(self.config.Generator, "res_scale", 0.2),
            )

        else:
            raise ValueError(f"Unknown generator model type: {self.config.Generator.model_type}")  # safety check

        if mode == "train": # only get discriminator in training mode
            # ======================================================================
            # SECTION: Initialize Discriminator
            # Purpose: Build discriminator network for adversarial training.
            # ======================================================================
            discriminator_type = getattr(self.config.Discriminator, 'model_type', 'standard')
            n_blocks = getattr(self.config.Discriminator, 'n_blocks', None)

            if discriminator_type == 'standard':
                from opensr_srgan.model.discriminators.srgan_discriminator import Discriminator

                discriminator_kwargs = {
                    "in_channels": self.config.Model.in_bands,
                }
                if n_blocks is not None:
                    discriminator_kwargs["n_blocks"] = n_blocks

                self.discriminator = Discriminator(**discriminator_kwargs)
            elif discriminator_type == 'patchgan':
                from opensr_srgan.model.discriminators.patchgan import PatchGANDiscriminator

                patchgan_layers = n_blocks if n_blocks is not None else 3
                self.discriminator = PatchGANDiscriminator(
                    input_nc=self.config.Model.in_bands,
                    n_layers=patchgan_layers,
                )
            else:
                raise ValueError(f"Unknown discriminator model type: {discriminator_type}")

    def setup_lightning(self):
        """
        Check for Versioning and Set options accordingly.
        - For PL 2.x, set manual optimization for GAN training.
        """
        # Check for PL version - Define PL Hooks accordingly
        if self.pl_version >= (2,0,0):
            self.automatic_optimization = False  # manual optimization for PL 2.x
            # Set up Training Step
            from opensr_srgan.model.training_step_PL import training_step_PL2
            self._training_step_implementation = MethodType(training_step_PL2, self)
        elif self.pl_version < (2,0,0):
            assert self.automatic_optimization is True, "For PL <2.0, automatic_optimization must be True."
            # Set up Training Step
            from opensr_srgan.model.training_step_PL import training_step_PL1
            self._training_step_implementation = MethodType(training_step_PL1, self)
        else:
            raise RuntimeError(f"Unsupported PyTorch Lightning version: {pl.__version__}")

    def initialize_ema(self):
        ema_cfg = getattr(self.config.Training, "EMA", None)
        self.ema: ExponentialMovingAverage | None = None
        self._ema_update_after_step = 0
        self._ema_applied = False
        if ema_cfg is not None and getattr(ema_cfg, "enabled", False):
            ema_decay = float(getattr(ema_cfg, "decay", 0.999))
            ema_device = getattr(ema_cfg, "device", None)
            use_num_updates = bool(getattr(ema_cfg, "use_num_updates", True))
            self.ema = ExponentialMovingAverage(
                self.generator,
                decay=ema_decay,
                use_num_updates=use_num_updates,
            )
            self._ema_update_after_step = int(getattr(ema_cfg, "update_after_step", 0))

    def forward(self, lr_imgs):
        # perform generative step (LR → SR)
        sr_imgs = self.generator(lr_imgs)   # pass LR input through generator network
        return sr_imgs                      # return super-resolved output

    @torch.no_grad()
    def predict_step(self, lr_imgs):
        """
        Prediction for deployment:
        - Automatically detects whether normalization is needed.
        - If input range ≈ [0,1] → skip normalization.
        - If input range ≈ [0,10000] → apply normalization.
        - Handles inference, histogram matching, and denormalization.
        """
        assert self.generator.training is False, "Generator must be in eval mode for prediction."  # ensure eval mode
        lr_imgs = lr_imgs.to(self.device)  # move to device (GPU or CPU)

        # --- Check if normalization is needed ---
        lr_min, lr_max = lr_imgs.min().item(), lr_imgs.max().item()  # get value range
        if lr_max > 1.5:  # Sentinel-2 style raw reflectance → normalize
            lr_imgs = normalise_10k(lr_imgs, stage="norm")           # normalize to 0–1 range
            needs_normalization = True
        else:
            needs_normalization = False                                       # already normalized

        # --- Perform super-resolution (optionally using EMA weights) ---
        context = self.ema.average_parameters(self.generator) if self.ema is not None else nullcontext()
        with context:
            sr_imgs = self.generator(lr_imgs)                        # forward pass (SR prediction)

        # --- Histogram match SR to LR ---
        sr_imgs = histogram_match(lr_imgs, sr_imgs)                  # match distributions

        # --- Denormalize only if normalization was applied ---
        if needs_normalization:
            sr_imgs = normalise_10k(sr_imgs, stage="denorm")         # convert back to original scale

        # --- Move to CPU and return ---
        sr_imgs = sr_imgs.cpu().detach()                             # detach from graph for inference output
        return sr_imgs


    def training_step(self,batch,batch_idx, *args):
        # Check what we need to pass to the training function
        # Depending on PL version, and depending on the manual optimization
        if self.pl_version >= (2,0,0):
            return self._training_step_implementation(batch, batch_idx) # no optim_idx
        else:
            optimizer_idx = args[0] if len(args) > 0 else 0 # get optim_idx from kwargs
            return self._training_step_implementation(batch, batch_idx, optimizer_idx) # pass optim_idx
        
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=None,
        optimizer_closure=None,
        **kwargs,           # absorbs on_tpu/using_lbfgs/etc across PL versions
    ):
        """
        Used only when self.automatic_optimization == True (PL 1.x auto-optim).
        No-op for PL 2 manual because Lightning won't call it there.
        """
        # If we're in manual optimization (PL >=2 path), do nothing special.
        if not self.automatic_optimization:
            # Let Lightning's default behavior proceed (or simply return).
            # In manual mode we call opt.step()/zero_grad() in training_step_PL2.
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **kwargs)

        # ---- PL 1.x auto-optimization path ----
        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()
        optimizer.zero_grad()

        # EMA after the generator step (assumes G is optimizer_idx == 1)
        if (
            self.ema is not None
            and optimizer_idx == 1
            and self.global_step >= self._ema_update_after_step
        ):
            self.ema.update(self.generator)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # ======================================================================
        # SECTION: Forward pass — Generate SR prediction from LR input
        # Purpose: Run model inference on validation batch without gradient tracking.
        # ======================================================================
        """ 1. Extract and Predict """
        lr_imgs, hr_imgs = batch                            # unpack LR and HR tensors
        sr_imgs = self.forward(lr_imgs)                     # run generator to produce SR prediction

        # ======================================================================
        # SECTION: Compute and log validation metrics
        # Purpose: measure content-based metrics (PSNR/SSIM/etc.) on SR vs HR.
        # ======================================================================
        """ 2. Log Generator Metrics """
        metrics_hr_img = torch.clone(hr_imgs)               # clone to avoid in-place ops on autograd graph
        metrics_sr_img = torch.clone(sr_imgs)               # same for SR
        #metrics = calculate_metrics(metrics_sr_img, metrics_hr_img, phase="val_metrics")
        metrics = self.content_loss_criterion.return_metrics(
            metrics_sr_img, metrics_hr_img, prefix="val_metrics/"
        )                                                   # compute metrics using loss criterion helper
        del metrics_hr_img, metrics_sr_img                   # free cloned tensors from GPU memory

        for key, value in metrics.items():                   # iterate over metrics dict
            self.log(f"{key}", value,sync_dist=True)         # log each metric to logger (e.g., W&B, TensorBoard)

        # ======================================================================
        # SECTION: Optional visualization — Log example SR/HR/LR images
        # Purpose: visually track qualitative progress of the model.
        # ======================================================================
        # only perform image logging for first N batches to avoid logging all 200 images
        if batch_idx < self.config.Logging.num_val_images:
            base_lr = lr_imgs                                # use original LR for visualization

            # --- Select visualization bands (if multispectral) ---
            if self.config.Model.in_bands > 3:               # e.g., Sentinel-2 with >3 channels
                idx = np.random.choice(sr_imgs.shape[1], 3, replace=False)  # randomly select 3 bands
                lr_vis = base_lr[:, idx, :, :]               # subset LR
                hr_vis = hr_imgs[:, idx, :, :]               # subset HR
                sr_vis = sr_imgs[:, idx, :, :]               # subset SR
            else:
                lr_vis = base_lr                             # if RGB, just use all bands
                hr_vis = hr_imgs
                sr_vis = sr_imgs

            # --- Clone tensors for plotting to avoid affecting main tensors ---
            plot_lr_img = lr_vis.clone()
            plot_hr_img = hr_vis.clone()
            plot_sr_img = sr_vis.clone()

            # --- Generate matplotlib visualization (LR, SR, HR side-by-side) ---
            val_img = plot_tensors(plot_lr_img, plot_sr_img, plot_hr_img, title="Val")

            # --- Cleanup ---
            del plot_lr_img, plot_hr_img, plot_sr_img         # free memory after plotting

            # --- Log image to WandB (or compatible logger), if wanted ---
            if self.config.Logging.wandb.enabled:
                self.logger.experiment.log({"Val SR": wandb.Image(val_img)})  # upload to dashboard

            
            """ 3. Log Discriminator metrics """
            # If in pretraining, discard D metrics
            if self._pretrain_check(): # check if we'e in pretrain phase
                self.log("discriminator/adversarial_loss",
                        torch.zeros(1, device=lr_imgs.device),
                        prog_bar=False, sync_dist=True)
            else:
                # run discriminator and get loss between pred labels and true labels
                hr_discriminated = self.discriminator(hr_imgs)
                sr_discriminated = self.discriminator(sr_imgs)
                adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

                # Binary Cross-Entropy loss
                adversarial_loss = self.adversarial_loss_criterion(sr_discriminated,
                                                                torch.zeros_like(sr_discriminated)) + self.adversarial_loss_criterion(hr_discriminated,
                                                                                                                                        torch.ones_like(hr_discriminated))
                self.log("validation/DISC_adversarial_loss",adversarial_loss,sync_dist=True)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._apply_generator_ema_weights()

    def on_validation_epoch_end(self):
        self._restore_generator_weights()
        super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._apply_generator_ema_weights()

    def on_test_epoch_end(self):
        self._restore_generator_weights()
        super().on_test_epoch_end()

    def configure_optimizers(self):
        # configure Generator optimizer (Adam)
        optimizer_g = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.generator.parameters()),  # only trainable params
            lr=self.config.Optimizers.optim_g_lr                                     # LR from config
        )

        # configure Discriminator optimizer (Adam)
        optimizer_d = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),  # only trainable params
            lr=self.config.Optimizers.optim_d_lr                                       # LR from config
        )

        # learning rate schedulers (ReduceLROnPlateau)
        scheduler_g = ReduceLROnPlateau(
            optimizer_g, mode='min',
            factor=self.config.Schedulers.factor_g,
            patience=self.config.Schedulers.patience_g,
            #verbose=self.config.Schedulers.verbose
        )
        scheduler_d = ReduceLROnPlateau(
            optimizer_d, mode='min',
            factor=self.config.Schedulers.factor_d,
            patience=self.config.Schedulers.patience_d,
            #verbose=self.config.Schedulers.verbose
        )

        # optional generator warmup scheduler (step-based)
        warmup_steps = int(getattr(self.config.Schedulers, "g_warmup_steps", 0))
        warmup_type = getattr(self.config.Schedulers, "g_warmup_type", "none").lower()

        warmup_scheduler_config = None
        if warmup_steps > 0 and warmup_type in {"cosine", "linear"}:

            def _generator_warmup_lambda(current_step: int) -> float:
                if current_step >= warmup_steps:
                    return 1.0

                progress = (current_step + 1) / float(max(1, warmup_steps))
                if warmup_type == "linear":
                    return progress

                # default to cosine warmup for smoother start
                return 0.5 * (1.0 - math.cos(math.pi * progress))

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer_g,
                lr_lambda=_generator_warmup_lambda,
            )

            warmup_scheduler_config = {
                'scheduler': warmup_scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'generator_warmup',
            }

        scheduler_configs = [
            {'scheduler': scheduler_d, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1},
            {'scheduler': scheduler_g, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1}
        ]

        if warmup_scheduler_config is not None:
            scheduler_configs.append(warmup_scheduler_config)

        # return both optimizers + schedulers for PL
        return [
            [optimizer_d, optimizer_g],  # order super important, it's [D, G] and checked in training step
            scheduler_configs,
        ]
    
    
    def on_train_batch_start(self, batch, batch_idx):  # called before each training batch
        pre = self._pretrain_check()                   # check if currently in pretraining phase
        for p in self.discriminator.parameters():      # loop over all discriminator params
            p.requires_grad = not pre                  # freeze D during pretrain, unfreeze otherwise
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._log_lrs() # log LR's on each batch end

    def on_fit_start(self):  # called once at the start of training
        super().on_fit_start()
        if self.ema is not None and self.ema.device is None: # move ema weights
            self.ema.to(self.device)
            
            # ======================================================================
        # SECTION: Print Model Summary
        # Purpose: Output model architecture and parameter counts (only once).
        # ======================================================================
        from opensr_srgan.utils.gpu_rank import _is_global_zero
        if _is_global_zero():
            print_model_summary(self)  # print model summary to console

    def _log_generator_content_loss(self, content_loss: torch.Tensor) -> None:
        """Helper to consistently log the generator content loss across training phases."""
        self.log(
            "generator/content_loss",
            content_loss,
            prog_bar=True,
            sync_dist=True,
        )


    def _log_ema_setup_metrics(self) -> None:
        """Log static EMA configuration once training begins."""

        if getattr(self, "trainer", None) is None:
            return

        if self.ema is None:
            self.log(
                "EMA/enabled",
                0.0,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            return

        self.log(
            "EMA/enabled",
            1.0,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "EMA/decay",
            float(self.ema.decay),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "EMA/update_after_step",
            float(self._ema_update_after_step),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "EMA/use_num_updates",
            1.0 if self.ema.num_updates is not None else 0.0,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def _log_ema_step_metrics(self, *, updated: bool) -> None:
        """Log per-step EMA activity and statistics."""

        if self.ema is None:
            return

        self.log(
            "EMA/is_active",
            1.0 if updated else 0.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        steps_until_active = max(0, self._ema_update_after_step - self.global_step)
        self.log(
            "EMA/steps_until_activation",
            float(steps_until_active),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        if not updated:
            return

        self.log(
            "EMA/last_decay",
            float(self.ema.last_decay),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        if self.ema.num_updates is not None:
            self.log(
                "EMA/num_updates",
                float(self.ema.num_updates),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )


    def _pretrain_check(self):  # helper to check if still in pretrain phase
        if self.pretrain_g_only and self.global_step < self.g_pretrain_steps:  # true if pretraining active
            return True
        else:
            return False  # false once pretrain steps are exceeded

        
    def _compute_adv_loss_weight(self) -> float:
        """Compute the current adversarial loss weight using the configured ramp schedule."""
        beta = float(self.config.Training.Losses.adv_loss_beta)
        schedule = getattr(
            self.config.Training.Losses,
            "adv_loss_schedule",
            "cosine",
        ).lower()

        # Handle pretraining and edge cases early
        if self.global_step < self.g_pretrain_steps:
            return 0.0

        if self.adv_loss_ramp_steps <= 0 or self.global_step >= self.g_pretrain_steps + self.adv_loss_ramp_steps:
            return beta

        # Normalize progress to [0, 1]
        progress = (self.global_step - self.g_pretrain_steps) / self.adv_loss_ramp_steps
        progress = max(0.0, min(progress, 1.0))

        if schedule == "linear":
            return progress * beta

        if schedule == "cosine":
            # Cosine ramp to match the generator warmup behaviour
            return 0.5 * (1.0 - math.cos(math.pi * progress)) * beta

        raise ValueError(
            f"Unknown adversarial loss schedule '{schedule}'. Expected 'linear' or 'cosine'."
        )

    def _log_adv_loss_weight(self, adv_weight: float) -> None:
        """Log the current adversarial loss weight."""
        self.log("training/adv_loss_weight", adv_weight,sync_dist=True)

    def _adv_loss_weight(self):
        adv_weight = self._compute_adv_loss_weight()
        self._log_adv_loss_weight(adv_weight)
        return adv_weight

    def _apply_generator_ema_weights(self) -> None:
        if self.ema is None or self._ema_applied:
            return
        if self.ema.device is None:
            self.ema.to(self.device)
        self.ema.apply_to(self.generator)
        self._ema_applied = True

    def _restore_generator_weights(self) -> None:
        if self.ema is None or not self._ema_applied:
            return
        self.ema.restore(self.generator)
        self._ema_applied = False

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        if self.ema is not None and "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])

    def _log_lrs(self):
        # order matches your return: [optimizer_d, optimizer_g]
        opt_d = self.trainer.optimizers[0]
        opt_g = self.trainer.optimizers[1]
        self.log("lr_discriminator", opt_d.param_groups[0]["lr"],
                on_step=True, on_epoch=True, prog_bar=False, logger=True,sync_dist=True)
        self.log("lr_generator", opt_g.param_groups[0]["lr"],
                on_step=True, on_epoch=True, prog_bar=False, logger=True,sync_dist=True)
    
    def load_from_checkpoint(self,ckpt_path):
        # load ckpt
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['state_dict'])
        print(f"Loaded checkpoint from {ckpt_path}")
        

if __name__=="__main__":
    config_path = "opensr_srgan/configs/config_10m.yaml"
    model = SRGAN_model(config=str(config_path))
    model.forward(torch.randn(1,4,32,32))
    