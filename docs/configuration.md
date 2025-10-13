# Configuration Guide

Remote-Sensing-SRGAN is intentionally configuration-first. Every YAML file under `configs/` controls the entire experiment lifecycle — from dataset selection to optimiser schedules. This page documents the key sections and explains how they map to the Python implementation.

## Anatomy of a config file

Each config follows the same structure as `configs/config_10m.yaml`:

```yaml
Data:
  train_batch_size: 12
  val_batch_size: 8
  num_workers: 6
  prefetch_factor: 2
  dataset_type: "SISR_WW"

Model:
  in_bands: 6
  continue_training: False
  load_checkpoint: False

Training:
  pretrain_g_only: True
  g_pretrain_steps: 15000
  adv_loss_ramp_steps: 5000
  label_smoothing: True
  Losses:
    adv_loss_beta: 1e-3
    adv_loss_schedule: sigmoid
    l1_weight: 1.0
    sam_weight: 0.05
    perceptual_weight: 0.1
    perceptual_metric: vgg
    tv_weight: 0.0
    max_val: 1.0
    ssim_win: 11

Generator:
  model_type: cgan
  large_kernel_size: 9
  small_kernel_size: 3
  n_channels: 96
  n_blocks: 32
  scaling_factor: 8

Discriminator:
  model_type: standard
  n_blocks: 8

Optimizers:
  optim_g_lr: 1e-4
  optim_d_lr: 1e-4

Schedulers:
  metric: val_metrics/l1
  patience_g: 100
  patience_d: 100
  factor_g: 0.5
  factor_d: 0.5
  verbose: True

Logging:
  num_val_images: 5
```

## Data block

The data section controls loader throughput and selects the dataset implementation. The `dataset_type` key is forwarded to `data.data_utils.select_dataset`, which instantiates the proper dataset class and wraps it into a Lightning `DataModule` with the requested batch sizes and worker settings.【F:data/data_utils.py†L1-L95】【F:data/data_utils.py†L97-L150】

### Available dataset types

| Key | Description |
|-----|-------------|
| `S2_6b` | Loads Sentinel-2 SAFE chips with six 20 m bands (B05, B06, B07, B8A, B11, B12). Histogram-matched low-resolution inputs are synthesised on-the-fly. 【F:data/data_utils.py†L17-L48】|
| `S2_4b` | Reuses the SAFE pipeline for four 10 m bands (B05, B04, B03, B02). Useful for RGB+NIR 4× tasks. 【F:data/data_utils.py†L50-L82】|
| `SISR_WW` | Wraps the SEN2NAIP worldwide dataset for 4× cross-sensor training. Training/validation splits are constructed from a root directory. 【F:data/data_utils.py†L84-L95】|

If you introduce a new dataset class, register it by adding another branch to `select_dataset` and exposing a new `dataset_type` value.

## Model block

The `Model` section captures properties that affect both training and checkpoint management:

* `in_bands`: number of channels consumed by the generator and discriminator. This value is forwarded directly when models are instantiated.【F:model/SRGAN.py†L62-L100】
* `load_checkpoint`: path to a Lightning checkpoint whose weights should initialise the model before training begins.【F:train.py†L34-L47】
* `continue_training`: checkpoint to resume including optimiser states and scheduler counters (mapped to `Trainer(resume_from_checkpoint=...)`).【F:train.py†L34-L48】

## Training block

Training-related switches are consumed inside the Lightning module:

* `pretrain_g_only`: number of initial steps where only generator updates run (adversarial term disabled).【F:model/SRGAN.py†L34-L43】
* `g_pretrain_steps`: length of the generator-only warm-up window.【F:model/SRGAN.py†L34-L43】
* `adv_loss_ramp_steps`: number of iterations over which the adversarial loss weight ramps to its target. The schedule shape is controlled by `Losses.adv_loss_schedule`.【F:model/SRGAN.py†L34-L43】【F:configs/config_10m.yaml†L35-L70】
* `label_smoothing`: enables 0.9 real labels in the discriminator to reduce overconfidence.【F:model/SRGAN.py†L34-L43】

The nested `Losses` dictionary is passed to `GeneratorContentLoss`, which mixes pixel-space (`l1_weight`), spectral (`sam_weight`), perceptual (`perceptual_weight` with `perceptual_metric`), and total-variation (`tv_weight`) losses. The final adversarial weight after ramp-up is `adv_loss_beta`. 【F:model/SRGAN.py†L44-L58】【F:configs/config_10m.yaml†L35-L70】

## Generator and Discriminator blocks

Generator parameters control the backbone constructed in `SRGAN_model.get_models`:

* `model_type`: choose among `SRResNet`, `res`, `rcab`, `rrdb`, `lka`, or conditional GAN variants (`conditional_cgan`/`cgan`). Each path maps to a dedicated class under `model/generators/`.【F:model/SRGAN.py†L59-L101】
* `n_channels`, `n_blocks`: width and depth of the residual trunk. These values are forwarded verbatim to the generator constructors.【F:model/SRGAN.py†L64-L101】
* `scaling_factor`: upsampling factor (2×, 4×, or 8×). Passed into the generator to configure pixel-shuffle stages.【F:model/SRGAN.py†L64-L101】
* `large_kernel_size`, `small_kernel_size`: head/tail and residual block kernel sizes to shape receptive field.【F:model/SRGAN.py†L64-L101】

The discriminator section selects either the classic SRGAN CNN (`standard`) or a PatchGAN variant and optionally specifies convolutional depth via `n_blocks`. 【F:model/SRGAN.py†L102-L130】

## Optimisers and schedulers

Both the generator and discriminator use Adam optimisers with learning rates read from `Optimizers.optim_g_lr` and `Optimizers.optim_d_lr`. The Lightning module instantiates `ReduceLROnPlateau` schedulers using the parameters provided under `Schedulers` (`metric`, `patience_*`, `factor_*`, `verbose`).【F:model/SRGAN.py†L5-L12】【F:configs/config_10m.yaml†L103-L132】

## Logging

The `Logging` section controls qualitative outputs. During validation the model logs `num_val_images` panels via TensorBoard and Weights & Biases. The training script also wires in Weights & Biases, TensorBoard, and learning-rate monitors; edit this block if you want fewer images per epoch. 【F:configs/config_10m.yaml†L128-L132】【F:train.py†L59-L93】

## Tips for custom configs

* Keep configs under version control alongside experiment results — MkDocs can render them for quick reference.
* Use OmegaConf variable interpolation if you need to reuse values across sections.
* When experimenting with new datasets, add dataset-specific parameters (paths, band lists) under `Data` and consume them inside your dataset branch.
* Prefer creating new YAML files rather than editing defaults; the training script accepts any path via `--config`.

