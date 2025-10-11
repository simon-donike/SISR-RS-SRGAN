<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

# 🌍 Remote-Sensing-SRGAN

**Description:** **Remote-Sensing-SRGAN** is a flexible, research‑grade GAN framework for **super‑resolution (SR) of Sentinel‑2 and other remote‑sensing imagery**. It supports **arbitrary input band counts**, **configurable architectures**, **scalable depth/width**, and a **modular loss system**—with a robust training strategy (generator pretraining, adversarial ramp‑up, and discriminator schedules) that **stabilizes traditionally sensitive GAN training on EO data**.

---
## 🧑‍🚀 ToDOs  
- [ ] create validate.py  
- [ ] build interface with opensr-test  
- [ ] build interface with SEN2SR  
- [ ] incorporate the SEN2NAIP versions + downloading  
- [ ] make EVERYTHING flexible depending on config.yaml  
- [ ] implement different discriminators  
---

## 🧠 Overview

This repository provides:

* **Training code** for SRGAN‑style models tailored to remote sensing.
* A **flexible generator** with multiple block implementations and pluggable depths/widths.
* **Configurable losses** (content/perceptual/adversarial) with fully exposed **loss weights**.
* A **stabilized GAN procedure** (G‑only pretraining → adversarial ramp‑up → scheduled D updates) that makes RS‑SR training more reliable.
* Smooth integration with the **OpenSR** ecosystem for data handling, evaluation, and large‑scene inference.
* **Configuration‑first workflow**: everything — from generator/discriminator choices to loss weights and warmup length — is selectable in `configs/config.yaml`.

### Key Features

* 🧩 **Flexible generator**: choose block type `res`, `rcab`, `rrdb`, or `lka`; set `n_blocks`, `n_channels`, and `scale ∈ {2,4,8}`.
* 🛰️ **Flexible inputs**: train on **any band layout** (e.g., S2 RGB‑NIR, 6‑band stacks, or custom multispectral sets). Normalization/denorm utilities provided.
* ⚖️ **Flexible losses & weights**: combine L1, Spectral Angle Mapper, VGG19 or LPIPS perceptual distances, Total Variation, and a BCE adversarial term with **per‑loss weights**.
* 🧪 **Robust training strategy**: generator **pretraining**, **linear adversarial loss ramp**, and **discriminator update schedules/curves**.
* 📊 **Clear monitoring**: PSNR, SSIM, LPIPS, qualitative panels, and Weights & Biases logging.

---

## 🧱 Architectures & Blocks (short)

* **SRResNet (res)**: Residual blocks **without BN**, residual scaling; strong content backbone for pretraining.
* **RCAB (rcab)**: Residual Channel Attention Blocks (attention via channel‑wise reweighting) for enhanced detail contrast in textures.
* **RRDB (rrdb)**: Residual‑in‑Residual Dense Blocks (as in ESRGAN); deeper receptive fields with dense skip pathways for sharper detail.
* **LKA (lka)**: Large‑Kernel Attention blocks approximating wide‑context kernels; good for **large structures** common in RS (fields, roads, shorelines).

> All variants share the same I/O heads and upsampling (pixel‑shuffle) and can load compatible weights when shapes match.

## ⚙️ Config‑driven components

| Component | Options | Config keys |
|-----------|---------|-------------|
| **Generators** | `SRResNet`, `res`, `rcab`, `rrdb`, `lka` | `Generator.model_type`, depth via `Generator.n_blocks`, width via `Generator.n_channels`, kernels and scale. |
| **Discriminators** | `standard` SRGAN CNN, `patchgan` | `Discriminator.model_type`, granularity with `Discriminator.n_blocks`. |
| **Content losses** | L1, Spectral Angle Mapper, VGG19/LPIPS perceptual metrics, Total Variation | Weighted by `Training.Losses.*` (e.g. `l1_weight`, `sam_weight`, `perceptual_weight`, `perceptual_metric`, `tv_weight`). |
| **Adversarial loss** | BCE‑with‑logits on real/fake logits | Warmup via `Training.pretrain_g_only`, ramped by `adv_loss_ramp_steps`, capped at `adv_loss_beta`, optional label smoothing. |

The YAML keeps the SRGAN flexible: swap architectures or rebalance perceptual vs. spectral fidelity without touching the code.

---

## 🧰 Installation

```bash
# Clone the repository
git clone https://github.com/ESAOpenSR/Remote-Sensing-SRGAN.git
cd Remote-Sensing-SRGAN

# (optional) Create a virtual environment
python3 -m venv vsnv && source vsnv/bin/activate

# Install
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1) Generator Pretraining (content only)

Pretrain the generator (e.g., L1 + perceptual) to learn faithful reconstructions before adding GAN pressure.

```bash
python train.py --config configs/pretrain.yaml
```

### 2) Adversarial Finetuning (full GAN)

Enable the discriminator and ramp the adversarial term smoothly.

```bash
python train.py --config configs/srgan.yaml
```

### 3) Inference on Large Scenes

Use OpenSR‑Utils for tiled processing of SAFE/S2GM/GeoTIFF inputs.

```python
import opensr_utils
from opensr_utils.model_utils import get_srgan

model = get_srgan(weights="path/to/checkpoint.ckpt")
opensr_utils.large_file_processing(
    root="/path/to/S2_or_scene",
    model=model,
    output_dir="/path/to/output"
)
```

---

## 🏗️ Configuration Highlights

All key knobs are exposed via YAML:

* **Model**: `in_channels`, `n_channels`, `n_blocks`, `scale`, `block_type ∈ {SRResNet, res, rcab, rrdb, lka}`
* **Losses**: `l1_weight`, `sam_weight`, `perceptual_weight`, `tv_weight`, `adv_loss_beta`
* **Training**: `pretrain_g_only`, `g_pretrain_steps`, `adv_loss_ramp_steps`, `label_smoothing`, discriminator cadence controls
* **Data**: band order, normalization stats, crop sizes, augmentations

Example (excerpt):

```yaml
Model:
  in_channels: 6
  n_channels: 96
  n_blocks: 32
  scale: 4
  block_type: rcab
Training:
  pretrain_g_only: true
  g_pretrain_steps: 20000
  adv_loss_ramp_steps: 15000
  label_smoothing: true
Losses:
  l1_weight: 1.0
  sam_weight: 0.05
  perceptual_weight: 0.1
  adv_loss_beta: 0.001
```

---

## 🎚️ Training Strategy (stabilization)

* **G‑only pretraining:** Train with content/perceptual losses while the adversarial term is held at zero during the first `g_pretrain_steps`.
* **Adversarial ramp‑up:** Increase the BCE adversarial weight **linearly** over `adv_loss_ramp_steps` until it reaches `adv_loss_beta`.
* **Discriminator schedule:** Optionally update D with a **step curve** (e.g., 1:1, 1:2, or warm‑up skips) to avoid early D domination.

These choices are **purpose‑built for remote sensing**, where GANs are prone to hallucinations and optimization instabilities due to multi‑band inputs and domain shifts. The schedule and ramp make training **easier, safer, and more reproducible**.

---

## 🧪 Validation & Logging

* **Metrics:** PSNR, SSIM, LPIPS *(PSNR/SSIM use `sen2_stretch` with clipping for stable reflectance ranges)*
* **Visuals:** side‑by‑side LR/SR/HR panels (clamped, stretched), saved under `visualizations/`
* **W&B:** loss curves, example previews, system metrics
* **Outputs:** all logs, configs, and artifacts are centralized in `logs/`

---

## 🛰️ Datasets

Works with **Sentinel‑2** (10m/20m) and **other RS imagery** (e.g., Pleiades). Band‑flexible loaders and normalization are provided; integrate with **OpenSR‑Utils** for SAFE/S2GM/GeoTIFF ingestion and tiling.

---

## 📂 Repository Structure

```
Remote-Sensing-SRGAN/
├── models/                # Generator/Discriminator + block implementations
├── utils/                 # Normalization, stretching, plotting, logging
├── train.py               # Training entry point (Lightning-compatible)
├── validate.py            # Validation pipeline & visualizations
└── demo.py                # Minimal example
```

---

## 📚 Related Projects

* **OpenSR‑Model** – Latent Diffusion SR (LDSR‑S2)
* **OpenSR‑Utils** – Large‑scale inference & data plumbing
* **OpenSR‑Test** – Benchmarks & metrics
* **SEN2NEON** – Multispectral HR reference dataset

---

## ✍️ Citation

If you use this work, please cite:

```bibtex
coming soon...
```

---

## 🧑‍🚀 Authors & Acknowledgements

Developed by **Simon Donike** (IPL–UV) within the **ESA Φ‑lab / OpenSR** initiative. 

## Notes
This repo has been extensively reworked using Codex since I wanted to see if/how well it works. The AI changes wer concerned almost exclusively with structuring, commenting, and documentation. The GAN workflow itself was adapted from my previous implementations and the resulting experience with training these models: ([Remote-Sensing-SRGAN](https://github.com/simon-donike/Remote-Sensing-SRGAN)) and [NIR-GAN](https://github.com/simon-donike/NIR-GAN). The only exception is the loss object, even though its AI slop it works for now so I won't touch it again.