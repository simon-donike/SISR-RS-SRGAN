# SISR-RS-SRGAN

![Super-resolved Sentinel-2 example](assets/6band_banner.png)

SISR-RS-SRGAN is a comprehensive toolkit for training and evaluating super-resolution GANs on remote-sensing imagery. It
packages a flexible generator/discriminator zoo, composable perceptual and reconstruction losses, and the training heuristics
that make adversarial optimisation tractable—generator warm-up phases, learning-rate scheduling, adversarial-weight ramping, and
more. All options are driven by concise YAML configuration files so you can explore new architectures or datasets without
rewriting pipelines.

> This repository powers the GAN component of the ESA OpenSR project, delivering the reusable training core for its
> multispectral benchmarks.

## Why this repository?

* **One configuration, many models.** Swap between RCAN-style residual channel attention, RRDB, SRResNet, SwinIR-inspired
  backbones, and matching discriminators ranging from classic SRGAN to PatchGAN by editing a single config block.
* **Loss combinations that just work.** Mix pixel, perceptual, style, and adversarial objectives with sensible defaults for
  weights, schedules, and warm-up durations.
* **Battle-tested training loop.** PyTorch Lightning handles mixed precision, gradient accumulation, multi-GPU training, and
  restartable checkpoints while the repo layers in GAN-specific tweaks such as adversarial weight ramping and learning-rate
  restarts.
* **Remote-sensing aware defaults.** Normalisation, histogram matching, spectral-band handling, and Sentinel-2 SAFE ingestion are
  ready-made for 10 m and 20 m bands and easily extendable to other sensors.

## What you get out of the box

| Capability | Highlights |
| --- | --- |
| **Generators & discriminators** | RCAB, RRDB, residual-in-residual, large-kernel attention, PatchGAN, UNet-based discriminators, and more. |
| **Losses** | Weighted combinations of L1/L2, perceptual (VGG/LPIPS), style, and relativistic adversarial losses. |
| **Training utilities** | Generator warm-up phases, cosine and step learning-rate schedules, adversarial-weight ramping, EMA tracking, and mixed-precision support. |
| **Experiment management** | Configurable logging (Weights & Biases, TensorBoard), checkpointing, and experiment reproducibility hooks. |
| **Datasets** | Sentinel-2 SAFE archives, SEN2NAIP, and pluggable dataset interfaces for custom collections. |

## Repository tour

| Path | Description |
| --- | --- |
| `model/` | Lightning module, generator and discriminator implementations, and loss definitions. |
| `data/` | Dataset wrappers and helper utilities for Sentinel-2 SAFE archives and the SEN2NAIP world-wide corpus. |
| `configs/` | Ready-to-run YAML presets covering common scale factors, band selections, and architecture pairings. |
| `utils/` | Logging helpers, spectral normalisation utilities, and model summary functions used across the stack. |
| `train.py` | Command-line entry point that wires configuration, data module, loggers, and the Lightning trainer together. |

## Typical workflow

1. **Pick a configuration.** Start from a preset in `configs/` and adapt dataset paths, scale, generator, discriminator, and loss
   options to match your experiment.
2. **Prepare datasets.** Point the config to a Sentinel-2 SAFE manifest or the SEN2NAIP worldwide dataset and verify that the
   required bands exist on disk (see [Data](data.md)).
3. **Launch training.** Run `python train.py --config <path>` to instantiate the Lightning module, configure optimisers and
   callbacks, and start adversarial training (see [Training](training.md)).
4. **Monitor progress.** Use the included Weights & Biases and TensorBoard logging to track perceptual losses, adversarial
   metrics, and validation imagery.
5. **Deploy or evaluate.** The Lightning module exposes `predict_step` for batched inference, automatically normalising inputs and
   matching output histograms to the low-resolution source.

## Learn more

* [Architecture](architecture.md) explains how the Lightning module orchestrates generators, discriminators, and losses.
* [Configuration](configuration.md) documents every YAML field and how it influences training.
* [Data](data.md) details the supported datasets and how to integrate your own.
* [Getting started](getting-started.md) walks through environment setup and the first training run.
* [Training](training.md) covers logging, callbacks, and practical tips for stable optimisation.
* [Results](results.md) showcases ready-to-run configurations for common generator/discriminator pairs.

Whether you are reproducing published results or exploring new remote-sensing modalities, SISR-RS-SRGAN gives you a clear and
extensible foundation for multispectral super-resolution research.
