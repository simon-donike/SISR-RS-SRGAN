# Remote-Sensing SRGAN

Remote-Sensing-SRGAN is a research-grade training stack for single-image super-resolution (SISR) of multispectral satellite imagery. It wraps a flexible generator/discriminator design, configurable loss suite, and remote-sensing specific data pipelines behind a configuration-first workflow. The implementation is optimised for Sentinel-2 but can be adapted to other sensors that provide paired low-/high-resolution observations.

!!! info "Why another SRGAN?"
    Training GANs on multi-band Earth observation data is notoriously brittle. This project codifies the training heuristics that have proven stable in production — generator pretraining, adversarial weight ramp-up, and configurable discriminator cadence — while exposing every knob through YAML. The goal is to make it easy to reproduce remote-sensing SR experiments without rewriting boilerplate.

## Project highlights

### Flexible generator zoo

Choose between SRResNet, residual, RCAB, RRDB, large-kernel attention, or conditional GAN backbones with scale factors from 2×–8×.

```python
--8<-- "../model/SRGAN.py:lines=72-118"
```

### Pluggable losses

Pixel, spectral, perceptual, adversarial, and total-variation terms can be mixed with independent weights and activation schedules.

```python
--8<-- "../model/SRGAN.py:lines=34-58"
```

```yaml
--8<-- "../configs/config_10m.yaml:lines=35-70"
```

### Remote-sensing ready datasets

Sentinel-2 SAFE windowing and the SEN2NAIP worldwide pairs are built in, with Lightning datamodules created on the fly from the configuration.

```python
--8<-- "../data/data_utils.py:lines=1-95"
```

### Stabilised training flow

Generator-only warm-up, adversarial ramp-up, discriminator scheduling, and Lightning callbacks are wired into the training script.

```python
--8<-- "../train.py:lines=19-93"
```

### Comprehensive logging

TensorBoard visualisations, Weights & Biases tracking, and qualitative inspection panels are emitted during training.

```python
--8<-- "../model/SRGAN.py:lines=12-16"
```

```python
--8<-- "../train.py:lines=59-93"
```

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

