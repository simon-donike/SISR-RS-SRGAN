# Remote-Sensing SRGAN

Remote-Sensing-SRGAN is a research-grade training stack for single-image super-resolution (SISR) of multispectral satellite imagery. It wraps a flexible generator/discriminator design, configurable loss suite, and remote-sensing specific data pipelines behind a configuration-first workflow. The implementation is optimised for Sentinel-2 but can be adapted to other sensors that provide paired low-/high-resolution observations.

!!! info "Why another SRGAN?"
    Training GANs on multi-band Earth observation data is notoriously brittle. This project codifies the training heuristics that have proven stable in production — generator pretraining, adversarial weight ramp-up, and configurable discriminator cadence — while exposing every knob through YAML. The goal is to make it easy to reproduce remote-sensing SR experiments without rewriting boilerplate.

## Project highlights

* **Flexible generator zoo.** Choose between SRResNet, residual, RCAB, RRDB, large-kernel attention, or conditional GAN backbones with scale factors from 2×–8×.【F:model/SRGAN.py†L59-L101】
* **Pluggable losses.** Combine pixel, spectral, perceptual, adversarial, and total-variation terms with independent weights and activation schedules.【F:model/SRGAN.py†L44-L58】【F:configs/config_10m.yaml†L35-L70】
* **Remote-sensing ready datasets.** Sentinel-2 SAFE windowing and the SEN2NAIP worldwide pairs are built in, with Lightning datamodules created on the fly from the config.【F:data/data_utils.py†L1-L95】
* **Stabilised training flow.** Generator-only warm-up, adversarial ramp-up, discriminator scheduling, and Lightning callbacks are wired into the training script.【F:train.py†L19-L93】【F:model/SRGAN.py†L34-L43】
* **Comprehensive logging.** TensorBoard visualisations, Weights & Biases tracking, and qualitative inspection panels are emitted during training.【F:train.py†L59-L93】【F:model/SRGAN.py†L12-L16】

## Repository layout

```
SISR-RS-SRGAN/
├── configs/              # YAML experiment definitions
├── data/                 # Dataset implementations and helpers
├── model/                # LightningModule, generators, discriminators, losses
├── utils/                # Logging and spectral utilities
├── train.py              # Training entry point
└── docs/                 # MkDocs site (you are here)
```

## Next steps

* Head to [Getting Started](getting-started.md) for environment setup and the minimal training command.
* Review the [Configuration Guide](configuration.md) to understand every YAML switch.
* Dive into [Model Components](architecture.md) for generator and discriminator details.
* Explore [Data Pipelines](data.md) for dataset specifics and extension tips.

