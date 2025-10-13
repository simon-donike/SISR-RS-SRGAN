# Model Components

Remote-Sensing-SRGAN builds on a modular Lightning implementation that lets you swap generator and discriminator backbones as well as plug in new loss terms. This section describes the main components and how they interact during training.

## Lightning module

The central `SRGAN_model` class orchestrates the entire training loop:

* Loads configuration values from YAML via OmegaConf during initialisation.【F:model/SRGAN.py†L25-L43】
* Instantiates the chosen generator, discriminator, and perceptual loss network through `get_models()`.【F:model/SRGAN.py†L59-L150】
* Exposes `forward` for generator inference and `predict_step` for deployment-ready predictions with auto-normalisation and histogram matching.【F:model/SRGAN.py†L153-L192】
* Implements Lightning hooks for adversarial training, including generator-only pretraining, adversarial ramp-up, and rich logging (see `training_step` and helpers).【F:model/SRGAN.py†L195-L320】

## Generator zoo

Pick the generator backbone by setting `Generator.model_type` in the config. The options map to classes under `model/generators/`:

| Type | Description |
|------|-------------|
| `SRResNet` | Classic SRResNet with residual blocks sans batch norm; a strong baseline for content pretraining.【F:model/SRGAN.py†L64-L76】|
| `res` | Flexible residual blocks with configurable depth/width defined in `FlexibleGenerator` (supports residual scaling).【F:model/SRGAN.py†L77-L101】【F:model/generators/flexible_generator.py†L1-L96】|
| `rcab` | Residual channel attention blocks for finer texture modelling using the same flexible generator with RCAB building blocks.【F:model/generators/flexible_generator.py†L1-L96】【F:model/model_blocks/__init__.py†L144-L173】|
| `rrdb` | Residual-in-residual dense blocks (ESRGAN-style) for deeper receptive fields, enabled through the flexible generator registry.【F:model/generators/flexible_generator.py†L21-L96】【F:model/model_blocks/__init__.py†L176-L213】|
| `lka` | Large kernel attention variant approximating global receptive fields, useful for broad RS structures.【F:model/generators/flexible_generator.py†L21-L96】【F:model/model_blocks/__init__.py†L215-L249】|
| `conditional_cgan` / `cgan` | Conditional GAN generator that injects latent noise and conditional embeddings in addition to the spectral input.【F:model/SRGAN.py†L88-L101】【F:model/generators/cgan_generator.py†L15-L137】|

Each generator consumes `Model.in_bands` channels, expands to `Generator.n_channels` features, and upsamples by `Generator.scaling_factor` using pixel shuffle stages. Kernel sizes (`large_kernel_size`, `small_kernel_size`) tailor the receptive field for remote-sensing textures.【F:model/SRGAN.py†L64-L101】【F:model/generators/flexible_generator.py†L49-L96】

## Discriminators

Discriminator selection lives under `Discriminator.model_type`:

* `standard`: The canonical SRGAN discriminator with adjustable depth (`n_blocks`), implemented in `model/descriminators/srgan_discriminator.py`. The constructor accepts `in_channels` (matching the number of input bands).【F:model/SRGAN.py†L102-L117】
* `patchgan`: Patch-level discriminator inspired by pix2pix, parameterised by `n_layers` and automatically set from `n_blocks`. Good for local fidelity supervision on large tiles.【F:model/SRGAN.py†L118-L150】

Because both discriminators receive multi-band inputs, the repo avoids hard-coding RGB assumptions and allows training on arbitrary spectral stacks.

## Losses

Generator training minimises a combination of content and adversarial losses:

* **Content loss (`GeneratorContentLoss`)** combines weighted L1, spectral angle mapper (SAM), perceptual (VGG or LPIPS), and total variation terms as configured in the YAML file.【F:model/SRGAN.py†L34-L58】【F:model/loss/loss.py†L1-L210】
* **Adversarial loss** uses `torch.nn.BCEWithLogitsLoss` against discriminator logits with optional label smoothing to improve stability.【F:model/SRGAN.py†L34-L58】

During generator-only pretraining, the adversarial term is disabled; it ramps up to `adv_loss_beta` over `adv_loss_ramp_steps` iterations according to the chosen schedule (`linear` or `sigmoid`).【F:model/SRGAN.py†L34-L58】【F:configs/config_10m.yaml†L35-L70】

## Normalisation and inference helpers

Remote-sensing imagery often arrives as 0–10,000 scaled reflectance. `utils.spectral_helpers` provides `normalise_10k` for scaling to 0–1 and `histogram` for histogram matching. `predict_step` calls both utilities to ensure outputs match the statistical distribution of inputs before returning CPU tensors.【F:model/SRGAN.py†L160-L192】【F:utils/spectral_helpers.py†L53-L200】

## Logging utilities

Qualitative logging is handled via `utils.logging_helpers.plot_tensors`, which renders low-/super-/high-resolution panels for TensorBoard and Weights & Biases. The Lightning module invokes these helpers inside validation hooks so you can monitor spectral fidelity and artefacts during training.【F:model/SRGAN.py†L12-L16】【F:utils/logging_helpers.py†L1-L200】

## Extending the model

* Add new generator families by creating a file under `model/generators/` and wiring it into `SRGAN_model.get_models` plus `model/generators/__init__.py`.
* Introduce alternative discriminators under `model/descriminators/` and add a new branch in `get_models`.
* Extend content loss by editing `model/loss/loss.py`; the `GeneratorContentLoss` class already parses weights from the config.
* For additional logging (e.g., metrics beyond PSNR/SSIM/LPIPS), modify the validation hooks inside `model/SRGAN.py`.

