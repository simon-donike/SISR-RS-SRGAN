[![PyPI](https://img.shields.io/pypi/v/opensr-srgan)](https://pypi.org/project/opensr-srgan/)
![Python](https://img.shields.io/pypi/pyversions/opensr-srgan)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)
[![Coverage](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN/branch/main/graph/badge.svg)](https://app.codecov.io/gh/simon-donike/SISR-RS-SRGAN)

<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

![banner](docs/assets/6band_banner.png)

# 🌍 Single Image Super-Resolution Remote Sensing 'SRGAN'

**Remote-Sensing-SRGAN** is a research-grade GAN framework for **super-resolution of Sentinel-2 and other remote-sensing imagery**. It supports arbitrary band counts, configurable generator/discriminator designs, scalable depth/width, and a modular loss system designed for stable GAN training on EO data.

---

## 📖 Documentation

Full docs live at **[srgan.opensr.eu](https://www.srgan.opensr.eu/)**. They cover usage, configuration, training recipes, and deployment tips in depth.

## 🧠 Highlights

* **Flexible models:** swap between SRResNet, RCAB, RRDB, and LKA-style generators with YAML-only changes.
* **Remote-sensing aware losses:** combine spectral, perceptual, and adversarial objectives with tunable weights.
* **Stable training loop:** generator pretraining, adversarial ramp-ups, EMA, and multi-GPU Lightning support out of the box.
* **PyPI distribution:** `pip install opensr-srgan` for ready-to-use presets or custom configs.

---

## 🧱 Architectures & Blocks

See the [Architectures & Blocks guide](https://www.srgan.opensr.eu/model/architectures/) for supported generator/discriminator components, block variants, and configuration examples.

## 🧰 Installation

Follow the [installation instructions](https://www.srgan.opensr.eu/getting-started/installation/) for package, source, and dependency setup options.

---

## 🚀 Quickstart

* **Datasets:** Dataset structure, downloads, and custom loaders are documented in the [data guide](https://www.srgan.opensr.eu/data/).
* **Training:** Launch training with `python -m opensr_srgan.train --config opensr_srgan/configs/config.yaml` or follow the [training walkthrough](https://www.srgan.opensr.eu/getting-started/training/).
* **Inference:** Ready-made presets and large-scene pipelines are described in the [inference section](https://www.srgan.opensr.eu/getting-started/inference/).

---

## 🏗️ Configuration & Stabilization

All tunable knobs—architectures, loss weights, schedulers, and EMA—are exposed via YAML files under `opensr_srgan/configs`. Strategy tips for warm-ups, adversarial ramps, and EMA usage are summarised in the [training concepts chapter](https://www.srgan.opensr.eu/training/concepts/).
