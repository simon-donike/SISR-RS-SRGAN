[![PyPI](https://img.shields.io/pypi/v/opensr-srgan)](https://pypi.org/project/opensr-srgan/)
![Python](https://img.shields.io/pypi/pyversions/opensr-srgan)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)
[![Coverage](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN/branch/main/graph/badge.svg)](https://app.codecov.io/gh/simon-donike/SISR-RS-SRGAN)

<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

![banner](docs/assets/6band_banner.png)

# 🌍 Single Image Super-Resolution Remote Sensing "SRGAN"

**Remote-Sensing-SRGAN** is a research-grade GAN stack for **super-resolving Sentinel‑2 and other remote-sensing imagery**. It provides configurable generators/discriminators, modular loss functions, and a stabilised Lightning training loop that integrate with the wider OpenSR tooling.

Explore the full documentation at **[srgan.opensr.eu](https://www.srgan.opensr.eu/)**.

---

## 🧠 Overview

* Training code and configs for SRGAN-style models tailored to Earth observation.
* PyPI package (`opensr-srgan`) for drop-in inference helpers and configuration loading.
* Flexible architecture/loss options managed entirely through YAML configuration.
* Supports multi-band inputs, EMA stabilisation, multi-GPU training, and rich experiment logging.

---

## 🧱 Architectures & Blocks

See the [Architecture guide](https://srgan.opensr.eu/architecture/) for generator/discriminator variants, block diagrams, and configuration tips.

---

## 🧰 Installation

* **Package users:** `python -m pip install opensr-srgan`.
* **From source:** follow the [Getting started](https://srgan.opensr.eu/getting-started/#1-install-the-environment) guide for environment setup, optional dependencies, and CUDA-specific notes.

---

## 🚀 Quickstart

1. Copy or edit a config from `opensr_srgan/configs/`.
2. Launch training: `python -m opensr_srgan.train --config <path/to/config.yaml>`.
3. Monitor metrics/logs in Weights & Biases or TensorBoard.

Refer to [Training](https://srgan.opensr.eu/training/) and [Configuration](https://srgan.opensr.eu/configuration/) for step-by-step walkthroughs, CLI arguments, and logging options.

---

## 🏗️ Configuration Highlights

Key settings live in the YAML configs:

* **Model:** input bands, block type (`res`, `rcab`, `rrdb`, `lka`), scale factor.
* **Losses:** L1, SAM, perceptual (VGG/LPIPS), TV, adversarial weights.
* **Training:** pretraining length, adversarial ramp, scheduler warmups, EMA toggle.
* **Data:** dataset selector, band ordering, augmentations, crop sizes.

---

## 🎚️ Training Stabilisation

* Pretrain the generator before adversarial updates.
* Ramp adversarial weight and learning rates smoothly.
* Enable EMA weights for sharper validation results.

More detail is available in the [Training guide](https://srgan.opensr.eu/training/#stabilisation).

---

## 🧪 Validation & Logging

Default runs log PSNR/SSIM/LPIPS, discriminator statistics, and qualitative panels (locally in `logs/` and optionally to W&B). Custom logging hooks and presets are described in the [Results & logging](https://srgan.opensr.eu/results/) section of the docs.

---

## 🛰️ Datasets

Dataset setup instructions, manifest builders, and tips for extending the data loaders are documented in [Data](https://srgan.opensr.eu/data/).

---

## 📂 Repository Structure

```
SISR-RS-SRGAN/
├── opensr_srgan/         # Library + training code
├── docs/                 # MkDocs documentation sources
├── paper/                # Publication, figures, and supporting material
├── pyproject.toml        # Packaging metadata
└── requirements.txt      # Development dependencies
```

---

## 📚 Related Projects

* **OpenSR-Model** – Latent Diffusion SR (LDSR-S2)
* **OpenSR-Utils** – Large-scale inference & data plumbing
* **OpenSR-Test** – Benchmarks & metrics
* **SEN2NEON** – Multispectral HR reference dataset

---

## ✍️ Citation

If you use this work, please cite:

```bibtex
coming soon...
```

---

## 🧑‍🚀 Authors & Acknowledgements

Developed by **Simon Donike** (IPL–UV) within the **ESA Φ-lab / OpenSR** initiative.

---

## 🧑‍🚀 ToDOs
- [ ] create inference.py  (interface with opensr-test)
- [ ] build interface with SEN2SR (for 10m + 20m SR)
- [x] incorporate the SEN2NAIP versions + downloading
- [x] implement different discriminators
- [x] implement different visual loses (like LPIPS, VGG, ...)
- [ ] upgrade to torch>2.0 (complicated, PL doesnt support multiple schedulers in >2)
