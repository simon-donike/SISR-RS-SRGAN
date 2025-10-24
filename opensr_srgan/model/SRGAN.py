# Package Imports
import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Optional

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
    SRGAN_model
    ===========

    A flexible, PyTorch Lightning–based SRGAN implementation for single-image super-resolution
    in remote sensing and general imaging. The model supports multiple generator backbones
    (SRResNet, RCAB, RRDB, LKA, and a flexible registry-driven variant) and discriminator
    types (standard, PatchGAN), optional generator-only pretraining, adversarial loss ramp-up,
    and an Exponential Moving Average (EMA) of generator weights for more stable evaluation.

    Key features
    ------------
    - **Backbone flexibility:** Select generator/discriminator architectures via config.
    - **Training modes:** Generator pretraining, adversarial training, and LR warm-up.
    - **PL compatibility:** Automatic optimization for PL < 2.0; manual optimization for PL ≥ 2.0.
    - **EMA support:** Optional EMA tracking with delayed activation and device placement.
    - **Metrics & logging:** Content/perceptual metrics, LR logging, and optional W&B image logs.
    - **Inference helpers:** Normalization/denormalization for 0–10000 reflectance and histogram matching.

    Args
    ----
    config : str | pathlib.Path | dict | omegaconf.DictConfig, optional
        Path to a YAML file, an in-memory dict, or an OmegaConf config with sections
        defining Generator/Discriminator, Training, Optimizers, Schedulers, and Logging.
        Defaults to `"config.yaml"`.
    mode : {"train", "eval"}, optional
        Build both G and D in `"train"` mode; only G in `"eval"`. Defaults to `"train"`.

    Configuration overview (minimal)
    --------------------------------
    - **Model**: `in_bands` (int)
    - **Generator**: `model_type` (`"SRResNet"`, `"res"`, `"rcab"`, `"rrdb"`, `"lka"`, `"conditional_cgan"`), 
    `n_channels`, `n_blocks`, `large_kernel_size`, `small_kernel_size`, `scaling_factor`
    - **Discriminator**: `model_type` (`"standard"`, `"patchgan"`), `n_blocks` (optional)
    - **Training**:
    - `pretrain_g_only` (bool), `g_pretrain_steps` (int)
    - `adv_loss_ramp_steps` (int), `label_smoothing` (bool)
    - `Losses.adv_loss_beta` (float), `Losses.adv_loss_schedule` (`"linear"`|`"cosine"`)
    - `EMA.enabled` (bool), `EMA.decay` (float), `EMA.use_num_updates` (bool),
        `EMA.update_after_step` (int), `EMA.device` (str|None)
    - **Optimizers**: `optim_g_lr` (float), `optim_d_lr` (float)
    - **Schedulers**: `factor_g`, `factor_d`, `patience_g`, `patience_d`, `metric`,
    optional `g_warmup_steps`, `g_warmup_type` (`"linear"`|`"cosine"`)
    - **Logging**: `wandb.enabled` (bool), `num_val_images` (int)

    Behavior & versioning
    ---------------------
    - **PL ≥ 2.0**: Manual optimization (`automatic_optimization = False`). The bound
    `training_step_PL2` performs explicit `zero_grad/step` calls and handles EMA updates.
    - **PL < 2.0**: Automatic optimization. The legacy `training_step_PL1` is used, and
    `optimizer_step` coordinates stepping and EMA after generator updates.

    Created attributes (non-exhaustive)
    -----------------------------------
    generator : torch.nn.Module
        Super-resolution generator.
    discriminator : torch.nn.Module | None
        Only present in `"train"` mode.
    ema : ExponentialMovingAverage | None
        EMA tracker for the generator (if enabled).
    content_loss_criterion : torch.nn.Module
        Perceptual/pixel content loss wrapper used in training/validation.
    adversarial_loss_criterion : torch.nn.Module
        BCEWithLogits loss for adversarial training (D/G).

    Input/Output conventions
    ------------------------
    - **Forward**: `lr_imgs` of shape `(B, C, H, W)` → SR output with spatial scale set by
    `Generator.scaling_factor`.
    - **Predict**: Applies optional normalization (0–10000 → 0–1), EMA averaging, histogram
    matching, and denormalization back to the input range; returns CPU tensors.

    Example
    -------
    >>> model = SRGAN_model(config="config.yaml", mode="train")
    >>> # Trainer handles fit; inference via .predict_step() or forward():
    >>> model.eval()
    >>> with torch.no_grad():
    ...     sr = model(lr_imgs)

    Notes
    -----
    - Discriminator is frozen during generator pretraining (`pretrain_g_only=True` and
    `global_step < g_pretrain_steps`).
    - The adversarial loss contribution is ramped from 0 to `adv_loss_beta` over
    `adv_loss_ramp_steps` with linear or cosine schedule.
    - Learning rate warm-up for the generator is supported via a per-step LambdaLR.
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
        """Initialize and attach the Generator and (optionally) Discriminator models.

        This method builds the generator and discriminator architectures based on
        the configuration provided in `self.config`. It supports multiple generator
        backbones (e.g., SRResNet, RCAB, RRDB, LKA) and discriminator types
        (standard, PatchGAN). The discriminator is only initialized when the mode
        is set to `"train"`.

        Args:
            mode (str): Operational mode of the model. Must be one of:
                - `"train"`: Initializes both generator and discriminator.
                - Any other value: Initializes only the generator.

        Raises:
            ValueError: If an unknown generator or discriminator type is specified
                in the configuration.

        Attributes:
            generator (nn.Module): The initialized generator network instance.
            discriminator (nn.Module, optional): The initialized discriminator
                network instance (only present if `mode == "train"`).
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
        """Configure PyTorch Lightning behavior based on the detected version.

        This method ensures compatibility between different versions of
        PyTorch Lightning (PL) by setting appropriate optimization modes
        and binding the correct training step implementation.

        - For PL ≥ 2.0: Enables **manual optimization**, required for GAN training.
        - For PL < 2.0: Uses **automatic optimization** and the legacy training step.

        The selected training step function (`training_step_PL1` or `training_step_PL2`)
        is dynamically attached to the model as `_training_step_implementation`.

        Raises:
            AssertionError: If `automatic_optimization` is incorrectly set for PL < 2.0.
            RuntimeError: If the detected PyTorch Lightning version is unsupported.

        Attributes:
            automatic_optimization (bool): Indicates whether Lightning manages
                optimizer steps automatically.
            _training_step_implementation (Callable): Bound training step function
                corresponding to the active PL version.
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
        """Initialize the Exponential Moving Average (EMA) mechanism for the generator.

        This method sets up an EMA shadow copy of the generator parameters to
        stabilize training and improve the quality of generated outputs. EMA is
        enabled only if specified in the training configuration.

        The EMA model tracks the moving average of generator weights with a
        configurable decay factor and update schedule.

        Configuration fields under `config.Training.EMA`:
            - `enabled` (bool): Whether to enable EMA tracking.
            - `decay` (float): Exponential decay factor for weight averaging (default: 0.999).
            - `device` (str | None): Device to store the EMA weights on.
            - `use_num_updates` (bool): Whether to use step-based update counting.
            - `update_after_step` (int): Number of steps to wait before starting updates.

        Attributes:
            ema (ExponentialMovingAverage | None): EMA object tracking generator parameters.
            _ema_update_after_step (int): Step count threshold before EMA updates begin.
            _ema_applied (bool): Indicates whether EMA weights are currently applied to the generator.
        """
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
        """Forward pass through the generator network.

        Takes a batch of low-resolution (LR) input images and produces
        their corresponding super-resolved (SR) outputs using the generator model.

        Args:
            lr_imgs (torch.Tensor): Batch of input low-resolution images
                with shape `(B, C, H, W)` where:
                - `B`: batch size
                - `C`: number of channels
                - `H`, `W`: spatial dimensions.

        Returns:
            torch.Tensor: Super-resolved output images with increased spatial resolution,
            typically scaled by the model's configured upsampling factor.
        """
        sr_imgs = self.generator(lr_imgs)   # pass LR input through generator network
        return sr_imgs                      # return super-resolved output

    @torch.no_grad()
    def predict_step(self, lr_imgs):
        """Run a single super-resolution inference step.

        Performs forward inference using the generator (optionally under EMA weights)
        to produce super-resolved (SR) outputs from low-resolution (LR) inputs.
        The method automatically normalizes input values if required (e.g., raw
        Sentinel-2 reflectance), applies histogram matching, and denormalizes the
        outputs back to their original scale.

        Args:
            lr_imgs (torch.Tensor): Batch of input low-resolution images
                with shape `(B, C, H, W)`. Pixel value ranges may vary depending
                on preprocessing (e.g., 0–10000 for Sentinel-2 reflectance).

        Returns:
            torch.Tensor: Super-resolved output images with matched histograms
            and restored value range, detached from the computation graph and
            placed on CPU memory.

        Raises:
            AssertionError: If the generator is not in evaluation mode (`.eval()`).
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


    def training_step(self,batch,batch_idx, optimizer_idx: Optional[int] = None, *args):
        """Dispatch the correct training step implementation based on PyTorch Lightning version.

        This method acts as a compatibility layer between different PyTorch Lightning
        versions that handle multi-optimizer GAN training differently.

        - For PL ≥ 2.0: Manual optimization is used, and the optimizer index is not passed.
        - For PL < 2.0: Automatic optimization is used, and the optimizer index is passed
        to handle generator/discriminator updates separately.

        Args:
            batch (Any): A batch of training data (input tensors and targets as defined by the DataModule).
            batch_idx (int): Index of the current batch within the epoch.
            optimizer_idx (int | None, optional): Index of the active optimizer (0 for generator,
                1 for discriminator) when using PL < 2.0.
            *args: Additional arguments that may be passed by older Lightning versions.

        Returns:
            Any: The output of the active training step implementation, loss value.
        """
        # Depending on PL version, and depending on the manual optimization
        if self.pl_version >= (2,0,0):
            # In PL2.x, optimizer_idx is not passed, manual optimization is performed
            return self._training_step_implementation(batch, batch_idx) # no optim_idx
        else:
            # In Pl1.x, optimizer_idx arrives twice and is passed on
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
        """Custom optimizer step handling for PL 1.x automatic optimization.

        This method ensures correct behavior across different PyTorch Lightning
        versions and training modes. It is invoked automatically during training
        in PL < 2.0 when `automatic_optimization=True`. For PL ≥ 2.0, where manual
        optimization is used, this function is effectively bypassed.

        - In **PL ≥ 2.0 (manual optimization)**: The optimizer step is explicitly
        called within `training_step_PL2()`, including EMA updates.
        - In **PL < 2.0 (automatic optimization)**: This function manages optimizer
        stepping, gradient zeroing, and optional EMA updates after generator steps.

        Args:
            epoch (int): Current training epoch.
            batch_idx (int): Index of the current batch.
            optimizer (torch.optim.Optimizer): The active optimizer instance.
            optimizer_idx (int, optional): Index of the optimizer being stepped
                (e.g., 0 for discriminator, 1 for generator).
            optimizer_closure (Callable, optional): Closure for re-evaluating the
                model and loss before optimizer step (used with some optimizers).
            **kwargs: Additional arguments passed by PL depending on backend
                (e.g., TPU flags, LBFGS options).

        Notes:
            - EMA updates are performed only after generator steps (optimizer_idx == 1).
            - The update starts after `self._ema_update_after_step` global steps.

        """
        # If we're in manual optimization (PL >=2 path), do nothing special.
        if not self.automatic_optimization:
            # In manual mode we call opt.step()/zero_grad() in training_step_PL2.
            # In manual mode, we update EMA weights manually in training step too.
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
        """Run the validation loop for a single batch.

        This method performs super-resolution inference on validation data,
        computes image quality metrics (e.g., PSNR, SSIM), logs them, and
        optionally visualizes SR–HR–LR triplets. It also evaluates the
        discriminator’s adversarial response if applicable.

        Workflow:
            1. Forward pass (LR → SR) through the generator.
            2. Compute content-based validation metrics.
            3. Optionally log visual examples to the logger (e.g., Weights & Biases).
            4. Compute and log discriminator metrics, unless in pretraining mode.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple `(lr_imgs, hr_imgs)` of
                low-resolution and high-resolution tensors with shape `(B, C, H, W)`.
            batch_idx (int): Index of the current validation batch.

        Returns:
            None: Metrics and images are logged via Lightning’s logger interface.

        Raises:
            AssertionError: If an unexpected number of bands or invalid visualization
                configuration is encountered.

        Notes:
            - Validation is executed without gradient tracking.
            - Only the first `config.Logging.num_val_images` batches are visualized.
            - If EMA is enabled, the generator predictions reflect the current EMA state.
        """
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
            if self.config.Model.in_bands <3:
                # show only first band
                lr_vis = base_lr[:, :1, :, :]               # e.g., single-band input
                hr_vis = hr_imgs[:, :1, :, :]               # subset HR
                sr_vis = sr_imgs[:, :1, :, :]               # subset SR
            elif self.config.Model.in_bands == 3:
                # we can show normally
                pass
            elif self.config.Model.in_bands == 4:
                # assume its RGB-NIR, show RGB
                lr_vis = base_lr[:, :3, :, :]               # e.g., Sentinel-2 RGB
                hr_vis = hr_imgs[:, :3, :, :]               # subset HR
                sr_vis = sr_imgs[:, :3, :, :]               # subset SR
            elif self.config.Model.in_bands > 4:               # e.g., Sentinel-2 with >3 channels
                # random selection of bands
                idx = np.random.choice(sr_imgs.shape[1], 3, replace=False)  # randomly select 3 bands
                lr_vis = base_lr[:, idx, :, :]               # subset LR
                hr_vis = hr_imgs[:, idx, :, :]               # subset HR
                sr_vis = sr_imgs[:, idx, :, :]               # subset SR
            else:
                # should not happen
                pass

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
        """Hook executed at the start of each validation epoch.

        Applies the Exponential Moving Average (EMA) weights to the generator
        before running validation to ensure evaluation uses the smoothed model
        parameters.

        Notes:
            - Calls the parent hook via `super().on_validation_epoch_start()`.
            - Restores original weights at the end of validation.
        """
        super().on_validation_epoch_start()
        self._apply_generator_ema_weights()

    def on_validation_epoch_end(self):
        """Hook executed at the end of each validation epoch.

        Restores the generator’s original (non-EMA) weights after validation.
        Ensures subsequent training or testing uses up-to-date parameters.

        Notes:
            - Calls the parent hook via `super().on_validation_epoch_end()`.
        """
        self._restore_generator_weights()
        super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        """Hook executed at the start of each testing epoch.

        Applies the Exponential Moving Average (EMA) weights to the generator
        before running tests to ensure consistent evaluation with the
        smoothed model parameters.

        Notes:
            - Calls the parent hook via `super().on_test_epoch_start()`.
            - Restores original weights at the end of testing.
        """
        super().on_test_epoch_start()
        self._apply_generator_ema_weights()

    def on_test_epoch_end(self):
        """Hook executed at the end of each testing epoch.

        Restores the generator’s original (non-EMA) weights after testing.
        Ensures the model is reset to its latest training state.

        Notes:
            - Calls the parent hook via `super().on_test_epoch_end()`.
        """
        self._restore_generator_weights()
        super().on_test_epoch_end()

    def configure_optimizers(self):
        """Set up optimizers and learning rate schedulers for GAN training.

        Initializes separate Adam optimizers for the generator and discriminator,
        along with optional learning rate schedulers. Supports ReduceLROnPlateau
        schedulers for both models and an optional generator warmup scheduler
        (linear or cosine) for smoother training initialization.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
                A tuple containing:
                - A list of optimizers in order `[optimizer_d, optimizer_g]`
                (order is crucial for multi-optimizer training).
                - A list of scheduler configurations for PyTorch Lightning.

        Configuration Fields:
            config.Optimizers:
                - `optim_g_lr` (float): Learning rate for the generator.
                - `optim_d_lr` (float): Learning rate for the discriminator.

            config.Schedulers:
                - `factor_g`, `factor_d` (float): Multiplicative LR reduction factor.
                - `patience_g`, `patience_d` (int): Epochs to wait before LR reduction.
                - `metric` (str): Metric name to monitor for ReduceLROnPlateau.
                - `g_warmup_steps` (int, optional): Steps for warmup phase (default: 0).
                - `g_warmup_type` (str, optional): Type of warmup schedule (`"linear"` or `"cosine"`).

        Notes:
            - Uses `ReduceLROnPlateau` for both generator and discriminator.
            - Optionally applies step-based warmup for the generator learning rate.
            - The returned order `[D, G]` must match the order expected by the training step.
        """
        
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
        """Hook executed before each training batch.

        Freezes or unfreezes discriminator parameters depending on the
        current training phase. During pretraining, the discriminator is
        frozen to allow the generator to learn reconstruction without
        adversarial pressure.

        Args:
            batch (Any): The current batch of training data.
            batch_idx (int): Index of the current batch in the epoch.
        """
        pre = self._pretrain_check()                   # check if currently in pretraining phase
        for p in self.discriminator.parameters():      # loop over all discriminator params
            p.requires_grad = not pre                  # freeze D during pretrain, unfreeze otherwise
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Hook executed after each training batch.

        Logs the current learning rates for all active optimizers to
        the logger for monitoring and debugging purposes.

        Args:
            outputs (Any): Outputs returned by `training_step`.
            batch (Any): The batch of data processed.
            batch_idx (int): Index of the current batch in the epoch.
        """
        self._log_lrs() # log LR's on each batch end

    def on_fit_start(self):  # called once at the start of training
        """Hook executed once at the beginning of model fitting.

        Performs setup tasks that must occur before training starts:
        - Moves EMA weights to the correct device.
        - Prints a model summary (only from global rank 0 in DDP setups).

        Notes:
            - Calls `super().on_fit_start()` to preserve Lightning’s default behavior.
            - The model summary is only printed by the global zero process
            to avoid duplicated output in distributed training.
        """
        super().on_fit_start()
        if self.ema is not None and self.ema.device is None: # move ema weights
            self.ema.to(self.device)
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
        """Log static Exponential Moving Average (EMA) configuration parameters.

        Records whether EMA is enabled, along with its core hyperparameters
        (decay rate, activation delay, update mode). This information is
        logged once when training begins to help track model configuration.

        Notes:
            - If EMA is disabled, logs `"EMA/enabled" = 0.0`.
            - Called after the trainer is initialized to ensure logging context.
        """
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
        """Log dynamic EMA statistics during training.

        Tracks per-step EMA state, including whether an update occurred,
        how many steps remain until activation, and the most recent decay value.
        These metrics provide insight into EMA behavior over time.

        Args:
            updated (bool): Whether EMA weights were updated in the current step.

        Notes:
            - If EMA is disabled, this function exits without logging.
            - Logs include:
                - `"EMA/is_active"`: Indicates if EMA is currently updating.
                - `"EMA/steps_until_activation"`: Steps remaining before EMA starts updating.
                - `"EMA/last_decay"`: Latest applied decay value.
                - `"EMA/num_updates"`: Total number of EMA updates performed.
        """
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


    def _pretrain_check(self) -> bool:
        """Check whether the model is still in the generator pretraining phase.

        Returns:
            bool: True if the generator-only pretraining phase is active
            (i.e., `global_step` < `g_pretrain_steps`), otherwise False.

        Notes:
            - During pretraining, the discriminator is frozen and only the
            generator is updated.
        """
        if self.pretrain_g_only and self.global_step < self.g_pretrain_steps:  # true if pretraining active
            return True
        else:
            return False  # false once pretrain steps are exceeded

        
    def _compute_adv_loss_weight(self) -> float:
        """Compute the current adversarial loss weighting factor.

        Determines how strongly the adversarial loss contributes to the total
        generator loss, following a configurable ramp-up schedule. This helps
        stabilize early training by gradually increasing the influence of the
        discriminator.

        Returns:
            float: The current adversarial loss weight for the active step.

        Configuration Fields:
            config.Training.Losses:
                - `adv_loss_beta` (float): Maximum scaling factor for adversarial loss.
                - `adv_loss_schedule` (str): Type of ramp schedule (`"linear"` or `"cosine"`).

        Notes:
            - Returns `0.0` during pretraining steps (`global_step < g_pretrain_steps`).
            - After the ramp-up phase, the weight saturates at `beta`.
            - Cosine schedule provides a smoother ramp-up than linear.

        Raises:
            ValueError: If an unknown schedule type is provided in the configuration.
        """
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
        """Log the current adversarial loss weight.

        Args:
            adv_weight (float): Scalar multiplier applied to the adversarial loss term.
        """
        self.log("training/adv_loss_weight", adv_weight,sync_dist=True)

    def _adv_loss_weight(self) -> float:
        """Compute and log the current adversarial loss weight.

        Calls the internal scheduler/heuristic to obtain the adversarial loss weight,
        logs it, and returns the value.

        Returns:
            float: The computed adversarial loss weight for the current step/epoch.
        """
        adv_weight = self._compute_adv_loss_weight()
        self._log_adv_loss_weight(adv_weight)
        return adv_weight

    def _apply_generator_ema_weights(self) -> None:
        """Swap the generator's parameters to their EMA-smoothed counterparts.

        Applies EMA weights to the generator for evaluation (e.g., val/test). A no-op if
        EMA is disabled or already applied. Moves EMA to the correct device if needed.

        Notes:
            - Sets an internal flag to avoid double application during the same phase.
        """
        if self.ema is None or self._ema_applied:
            return
        if self.ema.device is None:
            self.ema.to(self.device)
        self.ema.apply_to(self.generator)
        self._ema_applied = True

    def _restore_generator_weights(self) -> None:
        """Restore the generator's original (non-EMA) parameters.

        Reverts the parameter swap performed by `_apply_generator_ema_weights()`.
        A no-op if EMA is disabled or not currently applied.

        Notes:
            - Clears the internal "applied" flag to enable future swaps.
        """
        if self.ema is None or not self._ema_applied:
            return
        self.ema.restore(self.generator)
        self._ema_applied = False

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Augment the checkpoint with EMA state, if available.

        Adds the EMA buffer/metadata to the checkpoint so that EMA can be restored upon load.

        Args:
            checkpoint (dict): Mutable checkpoint dictionary provided by Lightning.
        """
        super().on_save_checkpoint(checkpoint)
        if self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore EMA state from a checkpoint, if present.

        Args:
            checkpoint (dict): Checkpoint dictionary provided by Lightning containing
                model state and optional `"ema_state"` entry.
        """
        super().on_load_checkpoint(checkpoint)
        if self.ema is not None and "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])

    def _log_lrs(self) -> None:
        """Log learning rates for discriminator and generator optimizers.

        Notes:
            - Assumes optimizers are ordered as `[optimizer_d, optimizer_g]` in the trainer.
            - Logs both on-step and on-epoch for easier tracking.
        """
        # order matches your return: [optimizer_d, optimizer_g]
        opt_d = self.trainer.optimizers[0]
        opt_g = self.trainer.optimizers[1]
        self.log("lr_discriminator", opt_d.param_groups[0]["lr"],
                on_step=True, on_epoch=True, prog_bar=False, logger=True,sync_dist=True)
        self.log("lr_generator", opt_g.param_groups[0]["lr"],
                on_step=True, on_epoch=True, prog_bar=False, logger=True,sync_dist=True)
    
    def load_from_checkpoint(self,ckpt_path) -> None:
        """Load model weights from a PyTorch Lightning checkpoint file.

        Loads the `state_dict` from the given checkpoint and maps it to the current device.

        Args:
            ckpt_path (str | pathlib.Path): Path to the `.ckpt` file saved by Lightning.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            KeyError: If the checkpoint does not contain a `'state_dict'` entry.
        """
        # load ckpt
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt['state_dict'])
        print(f"Loaded checkpoint from {ckpt_path}")
        