# Package Imports
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

# local imports
from ..utils.logging_helpers import plot_tensors
from ..utils.model_descriptions import print_model_summary
from ..utils.spectral_helpers import histogram as histogram_match
from ..utils.spectral_helpers import normalise_10k
from .model_blocks import ExponentialMovingAverage


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

    def __init__(self, config_file_path="config.yaml", mode="train"):
        super(SRGAN_model, self).__init__()

        # ======================================================================
        # SECTION: Load Configuration
        # Purpose: Load and parse model/training hyperparameters from YAML file.
        # ======================================================================
        self.config = OmegaConf.load(config_file_path)  # load config file with OmegaConf
        assert mode in {"train", "eval"}, "Mode must be 'train' or 'eval'"  # validate mode
        self.mode = mode                                # store mode (train/eval)

        # --- Training settings ---
        self.pretrain_g_only = bool(getattr(self.config.Training, "pretrain_g_only", False))  # pretrain generator only (default False)
        self.g_pretrain_steps = int(getattr(self.config.Training, "g_pretrain_steps", 0))     # number of steps for G pretraining
        self.adv_loss_ramp_steps = int(getattr(self.config.Training, "adv_loss_ramp_steps", 20000))  # linear ramp-up steps for adversarial loss
        self.adv_target = 0.9 if getattr(self.config.Training, "label_smoothing", False) else 1.0    # use 0.9 if label smoothing enabled, else 1.0

        # ======================================================================
        # SECTION: Initialize Generator
        # Purpose: Build generator network depending on selected architecture.
        # ======================================================================
        self.get_models(mode=self.mode)  # dynamically builds and attaches generator + discriminator

        # ======================================================================
        # SECTION: Initialize EMA
        # Purpose: Optional exponential moving average (EMA) tracking for generator weights
        # ======================================================================
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

        # ======================================================================
        # SECTION: Define Loss Functions
        # Purpose: Configure generator content loss and discriminator adversarial loss.
        # ======================================================================
        if self.mode == "train":
            from .loss import GeneratorContentLoss
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
            from .generators.srresnet import Generator
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
            from .generators.flexible_generator import FlexibleGenerator
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
            from .generators import ConditionalGANGenerator

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
                from .discriminators.srgan_discriminator import Discriminator

                discriminator_kwargs = {
                    "in_channels": self.config.Model.in_bands,
                }
                if n_blocks is not None:
                    discriminator_kwargs["n_blocks"] = n_blocks

                self.discriminator = Discriminator(**discriminator_kwargs)
            elif discriminator_type == 'patchgan':
                from .discriminators.patchgan import PatchGANDiscriminator

                patchgan_layers = n_blocks if n_blocks is not None else 3
                self.discriminator = PatchGANDiscriminator(
                    input_nc=self.config.Model.in_bands,
                    n_layers=patchgan_layers,
                )
            else:
                raise ValueError(f"Unknown discriminator model type: {discriminator_type}")


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
            normalized = True
        else:
            normalized = False                                       # already normalized

        # --- Perform super-resolution (optionally using EMA weights) ---
        context = self.ema.average_parameters(self.generator) if self.ema is not None else nullcontext()
        with context:
            sr_imgs = self.generator(lr_imgs)                        # forward pass (SR prediction)

        # --- Histogram match SR to LR ---
        sr_imgs = histogram_match(lr_imgs, sr_imgs)                  # match distributions

        # --- Denormalize only if normalization was applied ---
        if normalized:
            sr_imgs = normalise_10k(sr_imgs, stage="denorm")         # convert back to original scale

        # --- Move to CPU and return ---
        sr_imgs = sr_imgs.cpu().detach()                             # detach from graph for inference output
        return sr_imgs


    def training_step(self,batch,batch_idx,optimizer_idx):
        # ======================================================================
        # SECTION: Forward pass + metric logging (no gradients for metrics)
        # Purpose: compute SR prediction, evaluate training metrics, log them.
        # ======================================================================

        # -------- CREATE SR DATA --------
        lr_imgs, hr_imgs = batch                                  # unpack LR/HR tensors from dataloader batch
        sr_imgs = self.forward(lr_imgs)                          # forward pass of the generator to produce SR from LR

        # ======================================================================
        # SECTION: Pretraining phase gate
        # Purpose: decide if we are in the content-only pretrain stage.
        # ======================================================================

        # -------- DETERMINE PRETRAINING --------
        pretrain_phase = self._pretrain_check()                  # check schedule: True => content-only pretraining
        if optimizer_idx == 1:  # log whether pretraining is active or not
            self.log("training/pretrain_phase", float(pretrain_phase), prog_bar=False,sync_dist=True)  # log once per G step to track phase state

        # ======================================================================
        # SECTION: Pretraining branch (delegated)
        # Purpose: during pretrain, only content loss for G and dummy logging for D.
        # ======================================================================

        # -------- IF PRETRAIN: delegate --------
        if pretrain_phase:
            # run pretrain step separately and return loss here
            return self.pretraining_training_step(lr_imgs=lr_imgs, hr_imgs=hr_imgs, sr_imgs=sr_imgs, optimizer_idx=optimizer_idx)  # delegate pretrain logic (no forward here)

        # ======================================================================
        # SECTION: Adversarial training — Discriminator step
        # Purpose: update D to distinguish HR (real) vs SR (fake).
        # ======================================================================

        # -------- Normal Train: Discriminator Step  --------
        if optimizer_idx==0:
            # run discriminator and get loss between pred labels and true labels
            hr_discriminated = self.discriminator(hr_imgs)       # D(real): logits for HR images
            sr_discriminated = self.discriminator(sr_imgs.detach()) # detach so G doesn’t get gradients from D’s step

            # targets
            real_target = torch.full_like(hr_discriminated, self.adv_target) # get labels/fuzzy labels
            fake_target = torch.zeros_like(sr_discriminated) # zeros, since generative prediction

            # Binary Cross-Entropy loss
            loss_real = self.adversarial_loss_criterion(hr_discriminated, real_target)   # BCEWithLogitsLoss for D(G(x))
            loss_fake = self.adversarial_loss_criterion(sr_discriminated, fake_target)  # BCEWithLogitsLoss for D(y)
            adversarial_loss = loss_real + loss_fake # Sum up losses
            self.log("discriminator/adversarial_loss",adversarial_loss,sync_dist=True) # log weighted loss

            # [LOG-B] Always log D opinions: real probs in normal training
            with torch.no_grad():
                d_real_prob = torch.sigmoid(hr_discriminated).mean()   # estimate mean real probability
                d_fake_prob = torch.sigmoid(sr_discriminated).mean()   # estimate mean fake probability
            self.log("discriminator/D(y)_prob", d_real_prob, prog_bar=True,sync_dist=True)      # log D(real) confidence
            self.log("discriminator/D(G(x))_prob", d_fake_prob, prog_bar=True,sync_dist=True)   # log D(fake) confidence

            # return weighted discriminator loss
            return adversarial_loss                                # PL will use this to step the D optimizer

        # ======================================================================
        # SECTION: Adversarial training — Generator step
        # Purpose: update G to minimize content loss + (weighted) adversarial loss.
        # ======================================================================

        # -------- Normal Train: Generator Step  --------
        if optimizer_idx==1:

            """ 1. Get VGG space loss """
            # encode images
            content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)   # perceptual/content criterion (e.g., VGG)
            self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
            for key, value in metrics.items():
                self.log(f"train_metrics/{key}", value,sync_dist=True)                             # log detailed metrics without extra forward passes


            """ 2. Get Discriminator Opinion and loss """
            # run discriminator and get loss between pred labels and true labels
            sr_discriminated = self.discriminator(sr_imgs)                             # D(SR): logits for generator outputs
            adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated)) # keep taargets 1.0 for G loss

            """ 3. Weight the losses"""
            adv_weight = self._adv_loss_weight() # get adversarial weight based on current step
            adversarial_loss_weighted = (adversarial_loss * adv_weight) # weight adversarial loss
            total_loss = content_loss + adversarial_loss_weighted # total content loss
            self.log("generator/total_loss",total_loss,sync_dist=True)                # log combined objective (content + λ_adv * adv)

            # return Generator loss
            return total_loss                                                         # PL will use this to step the G optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        if (
            self.ema is not None
            and optimizer_idx == 1
            and self.global_step >= self._ema_update_after_step
        ):
            self.ema.update(self.generator)

    def pretraining_training_step(self, *, lr_imgs, hr_imgs, sr_imgs, optimizer_idx):
        """
        Pretraining branch logic (no forward here):
        - G step (optimizer_idx==1): content-only loss
        - D step (optimizer_idx==0): dummy loss + zeroed logs so closures run
        """

        # ======================================================================
        # SECTION: Generator (G) pretraining step
        # Purpose: train only the generator using content loss, no adversarial part.
        # ======================================================================
        if optimizer_idx == 1:
            content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)  # compute perceptual/content loss (e.g., VGG or L1)
            self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
            for key, value in metrics.items():
                self.log(f"train_metrics/{key}", value,sync_dist=True)                               # reuse computed metrics for logging

            # Ensure adversarial weight is logged even when not used during pretraining
            adv_weight = self._compute_adv_loss_weight()
            self._log_adv_loss_weight(adv_weight)
            return content_loss                                                        # return loss for optimizer step (G only)

        # ======================================================================
        # SECTION: Discriminator (D) pretraining step
        # Purpose: no real training — just log zeros and return dummy loss to satisfy closure.
        # ======================================================================
        elif optimizer_idx == 0:
            device, dtype = hr_imgs.device, hr_imgs.dtype                              # get tensor device and dtype for consistency
            zero = torch.tensor(0.0, device=device, dtype=dtype)                       # define reusable zero tensor

            # --- Log dummy discriminator "opinions" (always zero during pretrain) ---
            self.log("discriminator/D(y)_prob",    zero, prog_bar=True,  sync_dist=True)  # fake real-prob (always 0)
            self.log("discriminator/D(G(x))_prob", zero, prog_bar=True,  sync_dist=True)  # fake fake-prob (always 0)

            # --- Create dummy scalar loss (ensures PL closure runs) ---
            dummy = torch.zeros((), device=device, dtype=dtype, requires_grad=True)    # dummy value with grad for optimizer compatibility
            self.log("discriminator/adversarial_loss", dummy, sync_dist=True)          # log dummy adversarial loss (always 0)
            return dummy                                                               # return dummy loss (keeps Lightning loop intact)


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
            self.log(f"{key}", value,sync_dist=True)                        # log each metric to logger (e.g., W&B, TensorBoard)

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

            # --- Log image to WandB (or compatible logger) ---
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
            verbose=self.config.Schedulers.verbose
        )
        scheduler_d = ReduceLROnPlateau(
            optimizer_d, mode='min',
            factor=self.config.Schedulers.factor_d,
            patience=self.config.Schedulers.patience_d,
            verbose=self.config.Schedulers.verbose
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
        from ..utils.gpu_rank import _is_global_zero
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
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config_20m.yaml"
    model = SRGAN_model(config_file_path=str(config_path))
    model.forward(torch.randn(1,6,32,32))
    
    model.load_from_checkpoint("logs/SRGAN_6bands/2025-10-11_23-53-20/last.ckpt")