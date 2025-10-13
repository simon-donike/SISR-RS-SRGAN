# Model Components

Remote-Sensing-SRGAN builds on a modular Lightning implementation that lets you swap generator and discriminator backbones as well as plug in new loss terms. This section describes the main components and how they interact during training.

## Lightning module

The central `SRGAN_model` class orchestrates the entire training loop.

### Initialisation

Configuration values are loaded through OmegaConf, losses are configured, and a model summary is printed for quick inspection.

```python
--8<-- "../model/SRGAN.py:lines=25-71"
```

### Model wiring

`get_models()` attaches the generator and discriminator variants requested in the YAML configuration.

```python
--8<-- "../model/SRGAN.py:lines=72-150"
```

### Inference helpers

`forward` and `predict_step` wrap generator inference with automatic normalisation and histogram matching so predictions align with Sentinel-2 statistics.

```python
--8<-- "../model/SRGAN.py:lines=151-192"
```

### Training hooks

Lightning hooks drive generator-only pretraining, adversarial ramp-up, and detailed metric logging.

```python
--8<-- "../model/SRGAN.py:lines=195-320"
```

## Generator zoo

Pick the generator backbone by setting `Generator.model_type` in the config. The options map to classes under `model/generators/`:

| Type | Description |
|------|-------------|
| `SRResNet` | Classic SRResNet with residual blocks sans batch norm; a strong baseline for content pretraining.|
| `res` | Flexible residual blocks with configurable depth/width defined in `FlexibleGenerator` (supports residual scaling).|
| `rcab` | Residual channel attention blocks for finer texture modelling using the same flexible generator with RCAB building blocks.|
| `rrdb` | Residual-in-residual dense blocks (ESRGAN-style) for deeper receptive fields, enabled through the flexible generator registry.|
| `lka` | Large kernel attention variant approximating global receptive fields, useful for broad RS structures.|
| `conditional_cgan` / `cgan` | Conditional GAN generator that injects latent noise and conditional embeddings in addition to the spectral input.|

Each generator consumes `Model.in_bands` channels, expands to `Generator.n_channels` features, and upsamples by `Generator.scaling_factor` using pixel shuffle stages. Kernel sizes (`large_kernel_size`, `small_kernel_size`) tailor the receptive field for remote-sensing textures.

```python
--8<-- "../model/SRGAN.py:lines=82-118"
```

```python
--8<-- "../model/generators/flexible_generator.py:lines=1-96"
```

Specialised blocks (RRDB, RCAB, large-kernel attention) are registered in `model/model_blocks` and consumed by the flexible generator.

```python
--8<-- "../model/model_blocks/__init__.py:lines=144-249"
```

Conditional GAN support routes through the dedicated generator class, which injects noise vectors and optional conditional embeddings.

```python
--8<-- "../model/generators/cgan_generator.py:lines=15-137"
```

## Discriminators

Discriminator selection lives under `Discriminator.model_type`. The Lightning module wires either the standard SRGAN CNN or a PatchGAN variant and forwards multi-band inputs without assuming RGB ordering.

```python
--8<-- "../model/SRGAN.py:lines=102-150"
```

The PatchGAN implementation and supporting building blocks live under `model/descriminators/`.

```python
--8<-- "../model/descriminators/srgan_discriminator.py:lines=1-71"
```

Because both discriminators receive multi-band inputs, the repo avoids hard-coding RGB assumptions and allows training on arbitrary spectral stacks.

## Losses

Generator training minimises a combination of content and adversarial losses. The Lightning module instantiates both criteria and scales them according to the YAML configuration.

```python
--8<-- "../model/SRGAN.py:lines=34-64"
```

`GeneratorContentLoss` mixes L1, spectral angle mapper, perceptual metrics, and total variation into a single callable.

```python
--8<-- "../model/loss/loss.py:lines=1-210"
```

Training warm-ups and adversarial ramp schedules are defined directly in the configuration.

```yaml
--8<-- "../configs/config_10m.yaml:lines=35-70"
```

## Normalisation and inference helpers

Remote-sensing imagery often arrives as 0–10,000 scaled reflectance. `utils.spectral_helpers` provides `normalise_10k` for scaling to 0–1 and `histogram` for histogram matching. `predict_step` calls both utilities to ensure outputs match the statistical distribution of inputs before returning CPU tensors.

```python
--8<-- "../model/SRGAN.py:lines=160-192"
```

```python
--8<-- "../utils/spectral_helpers.py:lines=53-200"
```

## Logging utilities

Qualitative logging is handled via `utils.logging_helpers.plot_tensors`, which renders low-/super-/high-resolution panels for TensorBoard and Weights & Biases. The Lightning module invokes these helpers inside validation hooks so you can monitor spectral fidelity and artefacts during training.

```python
--8<-- "../model/SRGAN.py:lines=12-16"
```

```python
--8<-- "../utils/logging_helpers.py:lines=1-72"
```

## Extending the model

* Add new generator families by creating a file under `model/generators/` and wiring it into `SRGAN_model.get_models` plus `model/generators/__init__.py`.
* Introduce alternative discriminators under `model/descriminators/` and add a new branch in `get_models`.
* Extend content loss by editing `model/loss/loss.py`; the `GeneratorContentLoss` class already parses weights from the config.
* For additional logging (e.g., metrics beyond PSNR/SSIM/LPIPS), modify the validation hooks inside `model/SRGAN.py`.

