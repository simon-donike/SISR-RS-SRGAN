# Configuration Guide

Remote-Sensing-SRGAN is intentionally configuration-first. Every YAML file under `configs/` controls the entire experiment lifecycle — from dataset selection to optimiser schedules. This page documents the key sections and explains how they map to the Python implementation.

## Anatomy of a config file

Each config follows the same structure as `configs/config_10m.yaml`:

```yaml
--8<-- "configs/config_10m.yaml:lines=12-132"
```

## Data block

The data section controls loader throughput and selects the dataset implementation. The `dataset_type` key is forwarded to `data.data_utils.select_dataset`, which instantiates the proper dataset class and wraps it into a Lightning `DataModule` with the requested batch sizes and worker settings.

```python
--8<-- "data/data_utils.py:lines=1-150"
```

### Available dataset types

| Key | Description |
|-----|-------------|
| `S2_6b` | Loads Sentinel-2 SAFE chips with six 20 m bands (B05, B06, B07, B8A, B11, B12). Histogram-matched low-resolution inputs are synthesised on-the-fly. |
| `S2_4b` | Reuses the SAFE pipeline for four 10 m bands (B05, B04, B03, B02). Useful for RGB+NIR 4× tasks. |
| `SISR_WW` | Wraps the SEN2NAIP worldwide dataset for 4× cross-sensor training. Training/validation splits are constructed from a root directory. |

If you introduce a new dataset class, register it by adding another branch to `select_dataset` and exposing a new `dataset_type` value.

## Model block

The `Model` section captures properties that affect both training and checkpoint management:

* `in_bands`: number of channels consumed by the generator and discriminator. This value is forwarded directly when models are instantiated.
* `load_checkpoint`: path to a Lightning checkpoint whose weights should initialise the model before training begins.
* `continue_training`: checkpoint to resume including optimiser states and scheduler counters (mapped to `Trainer(resume_from_checkpoint=...)`).

```python
--8<-- "model/SRGAN.py:lines=62-118"
```

```python
--8<-- "train.py:lines=34-48"
```

## Training block

Training-related switches are consumed inside the Lightning module:

* `pretrain_g_only`: number of initial steps where only generator updates run (adversarial term disabled).
* `g_pretrain_steps`: length of the generator-only warm-up window.
* `adv_loss_ramp_steps`: number of iterations over which the adversarial loss weight ramps to its target. The schedule shape is controlled by `Losses.adv_loss_schedule`.
* `label_smoothing`: enables 0.9 real labels in the discriminator to reduce overconfidence.

```python
--8<-- "model/SRGAN.py:lines=34-58"
```

The nested `Losses` dictionary is passed to `GeneratorContentLoss`, which mixes pixel-space (`l1_weight`), spectral (`sam_weight`), perceptual (`perceptual_weight` with `perceptual_metric`), and total-variation (`tv_weight`) losses. The final adversarial weight after ramp-up is `adv_loss_beta`.

```yaml
--8<-- "configs/config_10m.yaml:lines=35-70"
```

```python
--8<-- "model/loss/loss.py:lines=1-210"
```

## Generator and Discriminator blocks

Generator parameters control the backbone constructed in `SRGAN_model.get_models`:

* `model_type`: choose among `SRResNet`, `res`, `rcab`, `rrdb`, `lka`, or conditional GAN variants (`conditional_cgan`/`cgan`). Each path maps to a dedicated class under `model/generators/`.
* `n_channels`, `n_blocks`: width and depth of the residual trunk. These values are forwarded verbatim to the generator constructors.
* `scaling_factor`: upsampling factor (2×, 4×, or 8×). Passed into the generator to configure pixel-shuffle stages.
* `large_kernel_size`, `small_kernel_size`: head/tail and residual block kernel sizes to shape receptive field.

```python
--8<-- "model/SRGAN.py:lines=72-150"
```

The discriminator section selects either the classic SRGAN CNN (`standard`) or a PatchGAN variant and optionally specifies convolutional depth via `n_blocks`.

```python
--8<-- "model/descriminators/srgan_discriminator.py:lines=1-71"
```

## Optimisers and schedulers

Both the generator and discriminator use Adam optimisers with learning rates read from `Optimizers.optim_g_lr` and `Optimizers.optim_d_lr`. The Lightning module instantiates `ReduceLROnPlateau` schedulers using the parameters provided under `Schedulers` (`metric`, `patience_*`, `factor_*`, `verbose`).

```python
--8<-- "model/SRGAN.py:lines=408-443"
```

```yaml
--8<-- "configs/config_10m.yaml:lines=103-132"
```

## Logging

The `Logging` section controls qualitative outputs. During validation the model logs `num_val_images` panels via TensorBoard and Weights & Biases. The training script also wires in Weights & Biases, TensorBoard, and learning-rate monitors; edit this block if you want fewer images per epoch.

```yaml
--8<-- "configs/config_10m.yaml:lines=128-132"
```

```python
--8<-- "train.py:lines=59-93"
```

## Tips for custom configs

* Keep configs under version control alongside experiment results — MkDocs can render them for quick reference.
* Use OmegaConf variable interpolation if you need to reuse values across sections.
* When experimenting with new datasets, add dataset-specific parameters (paths, band lists) under `Data` and consume them inside your dataset branch.
* Prefer creating new YAML files rather than editing defaults; the training script accepts any path via `--config`.

