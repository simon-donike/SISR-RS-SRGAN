[![PyPI](https://img.shields.io/pypi/v/opensr-srgan)](https://pypi.org/project/opensr-srgan/)
![Python](https://img.shields.io/pypi/pyversions/opensr-srgan)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/simon-donike/SISR-RS-SRGAN/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://srgan.opensr.eu)
[![Coverage](https://codecov.io/gh/simon-donike/SISR-RS-SRGAN/branch/main/graph/badge.svg)](https://app.codecov.io/gh/simon-donike/SISR-RS-SRGAN)



<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

![banner](docs/assets/6band_banner.png)

# 🌍 Single Image Super-Resolution Remote Sensing 'SRGAN'

**Description:** **Remote-Sensing-SRGAN** is a flexible, research‑grade GAN framework for **super‑resolution (SR) of Sentinel‑2 and other remote‑sensing imagery**. It supports **arbitrary input band counts**, **configurable architectures**, **scalable depth/width**, and a **modular loss system**—with a robust training strategy (generator pretraining, adversarial ramp‑up, and discriminator schedules) that **stabilizes traditionally sensitive GAN training on EO data**.

---

## 📖 Documentation
*New*: [Documentation!](https://www.srgan.opensr.eu/)

## 🧠 Overview

This repository provides:

* **Training code** for SRGAN‑style models tailored to remote sensing.
* A **flexible generator and discriminator** with multiple block implementations and pluggable depths/widths.
* **Configurable losses** (content/perceptual/adversarial) with fully exposed **loss weights**.
* A **stabilized GAN procedure** (G‑only pretraining → adversarial ramp‑up → scheduled D , EMA weights) that makes RS‑SR training more reliable.
* Smooth integration with the **OpenSR** ecosystem for data handling, evaluation, and large‑scene inference.
* A **PyPI package (`opensr-srgan`)** with helpers to load models directly from configs or download ready-to-run inference presets from the Hugging Face Hub.
* **Configuration‑first workflow**: everything — from generator/discriminator choices to loss weights and warmup length — is selectable in `opensr_srgan/configs/config.yaml`.

### Key Features

* 🧩 **Flexible generator**: choose block type `res`, `rcab`, `rrdb`, or `lka`; set `n_blocks`, `n_channels`, and `scale ∈ {2,4,8}`.
* 🛰️ **Flexible inputs**: train on **any band layout** (e.g., S2 RGB‑NIR, 6‑band stacks, or custom multispectral sets). Normalization/denorm utilities provided.
* ⚖️ **Flexible losses & weights**: combine L1, Spectral Angle Mapper, VGG19 or LPIPS perceptual distances, Total Variation, and a BCE adversarial term with **per‑loss weights**.
* 🧪 **Robust training strategy**: generator **pretraining**, **linear adversarial loss ramp**, **cosine/linear LR warmup**, and **discriminator update schedules/curves**.
* ⚡ **Multi-GPU acceleration**: run Lightning's DDP backend out of the box by listing multiple GPU IDs in `Training.gpus` for dramatically faster epochs on capable machines.
* 🌀 **Generator EMA tracking**: optional exponential moving average weights for sharper validation and inference results.
* 📊 **Clear monitoring**: PSNR, SSIM, LPIPS, qualitative panels, and Weights & Biases logging.

---

## 🧱 Architectures & Blocks (short)

* **SRResNet (res)**: Residual blocks **without BN**, residual scaling; strong content backbone for pretraining.
* **RCAB (rcab)**: Residual Channel Attention Blocks (attention via channel‑wise reweighting) for enhanced detail contrast in textures.
* **RRDB (rrdb)**: Residual‑in‑Residual Dense Blocks (as in ESRGAN); deeper receptive fields with dense skip pathways for sharper detail.
* **LKA (lka)**: Large‑Kernel Attention blocks approximating wide‑context kernels; good for **large structures** common in RS (fields, roads, shorelines).

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

### Option 1 — install the packaged model (recommended for inference)

The project can be consumed directly from [PyPI](https://pypi.org/project/opensr-srgan/):

```bash
python -m pip install opensr-srgan
```

After installation you have two options for model creation:

1. **Instantiate directly from a config + weights** when you manage checkpoints yourself.

   ```python
   from opensr_srgan import load_from_config

   model = load_from_config(
       config_path="opensr_srgan/configs/config_10m.yaml",
       checkpoint_uri="https://example.com/checkpoints/srgan.ckpt",
       map_location="cpu",  # optional
   )
   ```

2. **Load the packaged inference presets** (either `"RGB-NIR"` or `"SWIR"`).

   The helper fetches the appropriate configuration (e.g., `config_RGB-NIR.yaml`)
   and pretrained checkpoint (e.g., `RGB-NIR_4band_inference.ckpt`) from the
   [`simon-donike/SR-GAN`](https://huggingface.co/simon-donike/SR-GAN) repository
   on the Hugging Face Hub and caches them locally for reuse.

   ```python
   from opensr_srgan import load_inference_model

   rgb_model = load_inference_model("RGB-NIR", map_location="cpu")
   swir_model = load_inference_model("SWIR")
   ```

Both helpers return a ready-to-use `pytorch_lightning.LightningModule`; access
its `.generator` attribute for inference-ready PyTorch modules.

### Option 2 — work from source

> ⚠️ **Python version**: the pinned `torch==1.13.1` and `torchvision==0.14.1` wheels target
> Python 3.10 (or earlier). Create your environment with a Python 3.10 interpreter to avoid
> installation failures on newer runtimes (e.g., Python 3.11).

```bash
# Clone the repository
git clone https://github.com/ESAOpenSR/Remote-Sensing-SRGAN.git
cd Remote-Sensing-SRGAN

# (optional) Create a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# (recommended) Upgrade pip so dependency resolution succeeds
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# (optional) Install extras for LPIPS metrics or TacoReader data loading
# pip install lpips tacoreader
```

> ℹ️ **Tip:** If the default PyPI index cannot find `torch==1.13.1`, install
> PyTorch directly from the official wheel index before running
> `pip install -r requirements.txt`:
>
> ```bash
> # CUDA 11.7 builds
> pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
> ```

---

## 🚀 Quickstart

### 0) Data

Make sure the data folders exist and are correctly associated with the dataset classes in the dataset folder. Use either your own data or any of the provided datasets in the `opensr_srgan/data/` folder.

### 1) SRGAN Training

Train the GAN model.

```bash
python -m opensr_srgan.train --config opensr_srgan/configs/config.yaml
```

Multi-GPU training is enabled by setting `Training.gpus` in your config to a list of device indices (e.g. `[0, 1, 2, 3]`). The trainer automatically switches to Distributed Data Parallel (DDP), yielding significantly faster wall-clock times when scaling out across multiple GPUs.

### 2) Inference on Large Scenes

Use OpenSR‑Utils for tiled processing of SAFE/GeoTIFF inputs. If you installed the `opensr-srgan` package from PyPI you
can swap in the packaged helpers to obtain weights either from a local config/ckpt pair or directly from the Hugging Face
Hub presets.

For use-cases like these, we provide presets with the necessary `config.yaml` and weights that get pulled from HuggingFace.

```python
import opensr_utils
from opensr_srgan import load_inference_model

model = load_inference_model("RGB-NIR") # loads preset model straight from HF incl. weights
opensr_utils.large_file_processing(
    root="/path/to/S2_or_scene",
    model=model,
    output_dir="/path/to/output"
)
```

---

## 🏗️ Configuration Highlights

All key knobs are exposed via YAML in the `opensr_srgan/configs` folder:

* **Model**: `in_channels`, `n_channels`, `n_blocks`, `scale`, `block_type ∈ {SRResNet, res, rcab, rrdb, lka}`
* **Losses**: `l1_weight`, `sam_weight`, `perceptual_weight`, `tv_weight`, `adv_loss_beta`
* **Training**: `pretrain_g_only`, `g_pretrain_steps`, `adv_loss_ramp_steps`, `label_smoothing`, generator LR warmup (`Schedulers.g_warmup_steps`, `Schedulers.g_warmup_type`), discriminator cadence controls
* **Data**: band order, normalization stats, crop sizes, augmentations

---

## 🎚️ Training Stabilization Strategies

* **G‑only pretraining:** Train with content/perceptual losses while the adversarial term is held at zero during the first `g_pretrain_steps`.
* **Adversarial ramp‑up:** Increase the BCE adversarial weight **linearly** or smoothly (**cosine**) over `adv_loss_ramp_steps` until it reaches `adv_loss_beta`.
* **Generator LR warmup:** Ramp the generator optimiser with a **cosine** or **linear** schedule for the first 1–5k steps via `Schedulers.g_warmup_steps`/`g_warmup_type` before switching to plateau-based reductions.
* **EMA smoothing:** Enable `Training.EMA.enabled` to keep a shadow copy of the generator. Decay values in the 0.995–0.9999 range balance responsiveness with stability and are swapped in automatically for validation/inference.

The schedule and ramp make training **easier, safer, and more reproducible**.

---

## 🧪 Validation & Logging

* **Metrics:** PSNR, SSIM, LPIPS *(PSNR/SSIM use `sen2_stretch` with clipping for stable reflectance ranges)*
* **Visuals:** side‑by‑side LR/SR/HR panels (clamped, stretched), saved under `visualizations/`
* **W&B:** loss curves, example previews, system metrics
* **Outputs:** all logs, configs, and artifacts are centralized in `logs/` and on WandB.

---

## 🛰️ Datasets

Two dataset pipelines ship with the repository under `opensr_srgan/data/`. Both return `(lr, hr)` pairs that are wired into the training `LightningDataModule` through `opensr_srgan/data/data_utils.py`.

### Sentinel‑2 SAFE windowed chips

* **Purpose.** Allows training directly from raw Sentinel‑2 Level‑1C/Level‑2A `.SAFE` products. A manifest builder enumerates the granule imagery, records chip windows, and the dataset turns each window into an `(lr, hr)` pair.
* **Pipeline.**
  1. `S2SAFEWindowIndexBuilder` crawls a root directory of `.SAFE` products, collects the band metadata, and (optionally) windows each raster into fixed chip sizes, storing the results as JSON.
  2. `S2SAFEDataset` groups those single‑band windows by granule, stacks the requested band order, and crops everything to the requested high‑resolution size (default `512×512`).
  3. The stacked HR tensor is downsampled in code with anti‑aliased bilinear interpolation to create the LR observation, so the model sees the interpolated image as input and the original Sentinel‑2 patch as target. Invalid chips (NaNs, nodata, near‑black) are filtered out during training.
* **Setup.**
  1. Organise your `.SAFE` products under a common root (the builder expects the usual `GRANULE/<id>/IMG_DATA` structure).
  2. Run the builder (see the `__main__` example in `opensr_srgan/data/SEN2_SAFE/S2_6b_ds.py`) to generate a manifest JSON containing file metadata and chip coordinates.
  3. Instantiate `S2SAFEDataset` with the manifest path, the band list/order, your desired `hr_size`, and the super‑resolution factor. The dataset will normalise values and synthesise the LR input automatically.

### SEN2NAIP (4× Sentinel‑2 → NAIP pairs)

* **Purpose.** Wraps the Taco Foundation `SEN2NAIPv2` release, which provides pre‑aligned Sentinel‑2 observations and NAIP aerial reference chips. The dataset class simply reads the file paths stored in the `.taco` manifest and loads the rasters on the fly—Sentinel‑2 frames act as the low‑resolution input, NAIP tiles are the 4× higher‑resolution target.
* **Scale.** This loader is hard‑coded for 4× super‑resolution. The Taco manifest already contains the bilinearly downsampled Sentinel‑2 inputs, so no alternative scale factors are exposed.
* **Setup.**
  1. Install the optional dependencies used by the loader: `pip install tacoreader rasterio` (plus Git LFS for the download step).
  2. Fetch the dataset by running `python -m opensr_srgan.data.SEN2NAIP.download_S2N`. The helper script downloads the manifest and image tiles from the Hugging Face hub into the working directory.
  3. Point your config to the resulting `.taco` file when you instantiate `SEN2NAIP` (e.g. in a custom `select_dataset` branch). No extra preprocessing is required—the dataset returns NumPy arrays that are subsequently converted to tensors by the training pipeline.


### Adding a new dataset

1. **Create the dataset class** inside `opensr_srgan/data/<your_dataset>/`. Mirror the existing API (`__len__`, `__getitem__` returning `(lr, hr)`) so it can plug into the shared training utilities.
2. **Register it with the selector** by adding a new branch in `opensr_srgan/data/data_utils.py::select_dataset`, alongside the existing `S2_6b`/`S2_4b` options, so the configuration key resolves to your implementation.
3. **Expose a config toggle** by adding the new `Data.dataset_type` value to your experiment YAML (for example `opensr_srgan/configs/config_20m.yaml`). Point any dataset‑specific parameters (paths, band lists, scale factors) to your new loader inside that branch.

This keeps dataset plumbing centralised: dataset classes own their I/O logic, `select_dataset` wires them into Lightning, and the configuration file becomes the single switch for experiments.

---

## 📂 Repository Structure

```
SISR-RS-SRGAN/
├── opensr_srgan/
│   ├── configs/           # YAML presets for training and inference
│   ├── data/              # Dataset wrappers and download helpers
│   ├── model/             # Lightning module, generators, discriminators, losses
│   ├── utils/             # Logging, spectral helpers, model summaries
│   ├── train.py           # Training entry point (Lightning-compatible)
│   └── inference.py       # Convenience helpers for tiled inference
├── docs/                  # MkDocs documentation sources
├── paper/                 # Publication, figures, and supporting material
├── pyproject.toml         # Packaging metadata
└── requirements.txt       # Development dependencies
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

## 📒 Notes
This repo has been extensively reworked using Codex since I wanted to see if/how well it works. The AI changes were mostly about structuring, commenting, documentation, and small-scale features. The GAN workflow itself was adapted from my previous implementations and the resulting experience with training these models: ([Remote-Sensing-SRGAN](https://github.com/simon-donike/Remote-Sensing-SRGAN)) and [NIR-GAN](https://github.com/simon-donike/NIR-GAN).  
Only the SEN2 dataset class has been generated from scratch and can be considered AI slop. But since it works, I wont touch it again.

## 🧑‍🚀 ToDOs  
- [ ] create inference.py  (interface with opensr-test)
- [ ] build interface with SEN2SR (for 10m + 20m SR)
- [x] incorporate the SEN2NAIP versions + downloading  
- [x] implement different discriminators
- [x] implement different visual loses (like LPIPS, VGG, ...)
- [ ] upgrade to torch>2.0 (complicated, PL doesnt support multiple schedulers in >2)
