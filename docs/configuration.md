# Configuration

ESA OpenSR relies on YAML files to control every aspect of the training pipeline. This page documents the available keys and how they influence the underlying code. Use `opensr_srgan/configs/config_20m.yaml` and `opensr_srgan/configs/config_10m.yaml` as starting points.

## File structure

A typical configuration contains the following top-level sections:

```
Data:
Model:
Training:
Generator:
Discriminator:
Optimizers:
Schedulers:
Logging:
```

Each section maps directly to parameters consumed inside `opensr_srgan/model/SRGAN.py`, the dataset factory, or the training script.

## Data

| Key | Default | Description |
| --- | --- | --- |
| `train_batch_size` | 12 | Mini-batch size for the training dataloader. Falls back to `batch_size` if set. |
| `val_batch_size` | 8 | Batch size for validation. |
| `num_workers` | 6 | Number of worker processes for both dataloaders. |
| `prefetch_factor` | 2 | Additional batches prefetched by each worker. Ignored when `num_workers == 0`. |
| `dataset_type` | `ExampleDataset` | Dataset selector consumed by `opensr_srgan.data.dataset_selector.select_dataset`. |

## Model

| Key | Default | Description |
| --- | --- | --- |
| `in_bands` | 6 | Number of input channels expected by the generator and discriminator. |
| `continue_training` | `False` | Path to a Lightning checkpoint for resuming training (`Trainer.fit(resume_from_checkpoint=...)`). |
| `load_checkpoint` | `False` | Path to a checkpoint used solely for weight initialisation (no training state restored). |

## Training

### Warm-up and adversarial scheduling

| Key | Default | Description |
| --- | --- | --- |
| `pretrain_g_only` | `True` | Enable generator-only warm-up before adversarial updates. |
| `g_pretrain_steps` | `10000` | Number of optimiser steps spent in the warm-up phase. |
| `adv_loss_ramp_steps` | `5000` | Duration of the adversarial weight ramp after the warm-up. |
| `label_smoothing` | `True` | Replaces target value 1.0 with 0.9 for real examples to stabilise discriminator training. |

### Generator EMA (`Training.EMA`)

Maintaining an exponential moving average (EMA) of the generator smooths out sharp weight updates and usually yields sharper yet
stable validation imagery. The EMA is fully optional and controlled through the `Training.EMA` block:

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `False` | Turns EMA tracking on/off. When enabled, the EMA weights automatically replace the live generator during evaluation/inference. |
| `decay` | `0.999` | Smoothing factor applied at every update. Values closer to 1.0 retain longer history. |
| `update_after_step` | `0` | Defers EMA updates until the given optimiser step. Useful when you want the generator to warm up before tracking. |
| `device` | `null` | Stores EMA weights on a dedicated device (`"cpu"`, `"cuda:1"`, …). `null` keeps the weights on the same device as the generator. |
| `use_num_updates` | `True` | Enables PyTorch’s bias correction so the EMA ramps in smoothly during the first few updates. |

### Generator content loss (`Training.Losses`)

| Key | Default | Description |
| --- | --- | --- |
| `adv_loss_beta` | `1e-3` | Target weight applied to the adversarial term after ramp-up. |
| `adv_loss_schedule` | `cosine` | Ramp shape (`linear` or `cosine`). |
| `l1_weight` | `1.0` | Weight of the pixelwise L1 loss. |
| `sam_weight` | `0.05` | Weight of the spectral angle mapper loss. |
| `perceptual_weight` | `0.1` | Weight of the perceptual feature loss. |
| `perceptual_metric` | `vgg` | Backbone used for perceptual features (`vgg` or `lpips`). |
| `tv_weight` | `0.0` | Total variation regularisation strength. |
| `max_val` | `1.0` | Peak value assumed by PSNR/SSIM computations. |
| `ssim_win` | `11` | Window size for SSIM metrics. Must be an odd integer. |

## Generator

| Key | Default | Description |
| --- | --- | --- |
| `model_type` | `cgan` | Generator architecture (`SRResNet`, `res`, `rcab`, `rrdb`, `lka`, `conditional_cgan`, `cgan`). |
| `large_kernel_size` | `9` | Kernel size for input/output convolution layers. |
| `small_kernel_size` | `3` | Kernel size for residual/attention blocks. |
| `n_channels` | `96` | Base number of feature channels. |
| `n_blocks` | `32` | Number of residual/attention blocks. |
| `scaling_factor` | `8` | Super-resolution scale factor (2, 4, 8, ...). |

## Discriminator

| Key | Default | Description |
| --- | --- | --- |
| `model_type` | `standard` | Discriminator architecture (`standard` SRGAN or `patchgan`). |
| `n_blocks` | `8` | Number of convolutional blocks. PatchGAN defaults to 3 when unspecified. |

## Suggested settings

### Generator presets

The defaults in the YAML configs intentionally balance stability and fidelity for Sentinel-2 data. Start here before
performing sweeps:

* Keep `n_channels` around 96 for residual-style backbones so feature widths match the initial convolution used by the
  flexible generator factory.
* Depth drives detail. Begin with `n_blocks = 32` for flexible variants and reduce to 16 when training budgets are
  tight or when using the conditional generator, which already injects stochasticity via latent noise.
* Set `scaling_factor` according to your target resolution (2×/4×/8×); all bundled generators support those values out
  of the box.

| Generator type | Recommended `n_channels` | Recommended `n_blocks` | Typical `scaling_factor` | Notes |
| --- | --- | --- | --- | --- |
| `SRResNet` | 64 | 16 | 4× | Canonical baseline with batch-norm residual blocks; scale can be 2×/4×/8× as needed. |
| `res` | 96 | 32 | 4×–8× | Lightweight residual blocks without batch norm; works well for high-scale (8×) Sentinel data. |
| `rcab` | 96 | 32 | 4×–8× | Attention-enhanced residual blocks; keep depth high to exploit channel attention. |
| `rrdb` | 96 | 32 | 4×–8× | Dense residual blocks expand receptive field; expect higher VRAM use at 32 blocks. |
| `lka` | 96 | 24–32 | 4×–8× | Large-kernel attention blocks stabilise at moderate depth; drop to 24 blocks if memory bound. |
| `conditional_cgan`/`cgan` | 96 | 16 | 4× | Latent-modulated residual stack; pair with noise_dim≈128 and res_scale≈0.2 defaults. |

### Discriminator presets

Tune discriminator depth to match the generator capacity—too shallow and adversarial loss underfits, too deep and the training loop destabilises. These starting points mirror the architectures bundled with the repo:

| Discriminator type | Recommended depth parameter | Additional notes |
| --- | --- | --- |
| `standard` | `n_blocks = 8` | Mirrors the original SRGAN CNN with alternating stride-1/stride-2 blocks before the dense head.】 |
| `patchgan` | `n_blocks = 3` | Maps to the 3-layer PatchGAN (a.k.a. `n_layers`); increase to 4–5 for larger crops or when the generator is particularly sharp. |

When adjusting these presets, scale generator and discriminator together and monitor adversarial loss ramps defined in `Training.Losses` to keep training stable.

## Optimisers

| Key | Default | Description |
| --- | --- | --- |
| `optim_g_lr` | `1e-4` | Learning rate for the generator Adam optimiser. |
| `optim_d_lr` | `1e-4` | Learning rate for the discriminator Adam optimiser. |

## Schedulers

Both optimisers share the same configuration keys because they use `torch.optim.lr_scheduler.ReduceLROnPlateau`.

| Key | Default | Description |
| --- | --- | --- |
| `metric` | `val_metrics/l1` | Validation metric monitored for plateau detection. |
| `patience_g` | `100` | Epochs with no improvement before reducing the generator LR. |
| `patience_d` | `100` | Epochs with no improvement before reducing the discriminator LR. |
| `factor_g` | `0.5` | Multiplicative factor applied to the generator LR upon plateau. |
| `factor_d` | `0.5` | Multiplicative factor applied to the discriminator LR upon plateau. |
| `verbose` | `True` | Enables scheduler logging messages. |
| `g_warmup_steps` | `2000` | Number of optimiser steps used for generator LR warmup. Set to `0` to disable. |
| `g_warmup_type` | `cosine` | Warmup curve for the generator LR (`cosine` or `linear`). |

`g_warmup_steps` applies a step-wise warmup through `torch.optim.lr_scheduler.LambdaLR` before resuming the standard
`ReduceLROnPlateau` schedule. Cosine warmup is smoother for most runs, but a linear ramp (especially for 1–5k steps) remains
available for experiments that prefer a steady rise.

## Logging

| Key | Default | Description |
| --- | --- | --- |
| `num_val_images` | `5` | Number of validation batches visualised and logged to Weights & Biases each epoch. |

## Tips for managing configurations

* **Version control your YAML files.** Tracking them alongside experiment logs makes it easy to reproduce results.
* **Leverage OmegaConf interpolation.** You can reference other fields (e.g., reuse a base path) to avoid duplication.
* **Use descriptive filenames.** Include dataset, scale, and generator type in the config name to keep experiments organised.
* **Override selectively.** When launching through scripts or notebooks, you can load a base config and override specific fields at
  runtime using `OmegaConf.merge`.

With a clear understanding of these fields, you can rapidly iterate on architectures, datasets, and training strategies without modifying the underlying code.
