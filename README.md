<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

# 🌍 Remote-Sensing-SRGAN

**Description:** **Remote-Sensing-SRGAN** is a flexible, research‑grade GAN framework for **super‑resolution (SR) of Sentinel‑2 and other remote‑sensing imagery**. It supports **arbitrary input band counts**, **configurable architectures**, **scalable depth/width**, and a **modular loss system**—with a robust training strategy (generator pretraining, adversarial ramp‑up, and discriminator schedules) that **stabilizes traditionally sensitive GAN training on EO data**.

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

Make sure the datafolders exist and are correctly associated with the dataset classes in the dataset folder. Use either your own data or any of the provided dataset.

### 1) SRGAN Training

Train the GAN model.

```bash
python train.py --config configs/config.yaml
```

### 2) Inference on Large Scenes

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

Two dataset pipelines ship with the repository under `data/`. Both return `(lr, hr)` pairs that are wired into the training `LightningDataModule` through `data/data_utils.py`.

### SEN2NAIP (4× Sentinel‑2 → NAIP pairs)

* **Purpose.** Wraps the Taco Foundation `SEN2NAIPv2` release, which provides pre‑aligned Sentinel‑2 observations and NAIP aerial reference chips. The dataset class simply reads the file paths stored in the `.taco` manifest and loads the rasters on the fly—Sentinel‑2 frames act as the low‑resolution input, NAIP tiles are the 4× higher‑resolution target.
* **Scale.** This loader is hard‑coded for 4× super‑resolution. The Taco manifest already contains the bilinearly downsampled Sentinel‑2 inputs, so no alternative scale factors are exposed.
* **Setup.**
  1. Install the optional dependencies used by the loader: `pip install tacoreader rasterio` (plus Git LFS for the download step).
  2. Fetch the dataset by running `python data/SEN2AIP/download_S2N.py`. The helper script downloads the manifest and image tiles from the Hugging Face hub into the working directory.
  3. Point your config to the resulting `.taco` file when you instantiate `SEN2NAIP` (e.g. in a custom `select_dataset` branch). No extra preprocessing is required—the dataset returns NumPy arrays that are subsequently converted to tensors by the training pipeline.

### Sentinel‑2 SAFE windowed chips

* **Purpose.** Allows training directly from raw Sentinel‑2 Level‑1C/Level‑2A `.SAFE` products. A manifest builder enumerates the granule imagery, records chip windows, and the dataset turns each window into an `(lr, hr)` pair.
* **Pipeline.**
  1. `S2SAFEWindowIndexBuilder` crawls a root directory of `.SAFE` products, collects the band metadata, and (optionally) windows each raster into fixed chip sizes, storing the results as JSON.
  2. `S2SAFEDataset` groups those single‑band windows by granule, stacks the requested band order, and crops everything to the requested high‑resolution size (default `512×512`).
  3. The stacked HR tensor is downsampled in code with anti‑aliased bilinear interpolation to create the LR observation, so the model sees the interpolated image as input and the original Sentinel‑2 patch as target. Invalid chips (NaNs, nodata, near‑black) are filtered out during training.
* **Setup.**
  1. Organise your `.SAFE` products under a common root (the builder expects the usual `GRANULE/<id>/IMG_DATA` structure).
  2. Run the builder (see the `__main__` example in `data/SEN2_SAFE/S2_6b_ds.py`) to generate a manifest JSON containing file metadata and chip coordinates.
  3. Instantiate `S2SAFEDataset` with the manifest path, the band list/order, your desired `hr_size`, and the super‑resolution factor. The dataset will normalise values and synthesise the LR input automatically.

### Adding a new dataset

1. **Create the dataset class** inside `data/<your_dataset>/`. Mirror the existing API (`__len__`, `__getitem__` returning `(lr, hr)`) so it can plug into the shared training utilities.
2. **Register it with the selector** by adding a new branch in `data/data_utils.py::select_dataset`, alongside the existing `S2_6b`/`S2_4b` options, so the configuration key resolves to your implementation.
3. **Expose a config toggle** by adding the new `Data.dataset_type` value to your experiment YAML (for example `configs/config_20m.yaml`). Point any dataset‑specific parameters (paths, band lists, scale factors) to your new loader inside that branch.

This keeps dataset plumbing centralised: dataset classes own their I/O logic, `select_dataset` wires them into Lightning, and the configuration file becomes the single switch for experiments.

---

## 📂 Repository Structure

```
Remote-Sensing-SRGAN/
├── models/                # Generator/Discriminator + block implementations
├── utils/                 # Normalization, stretching, plotting, logging
├── utils/                 # Dataset implementations and downloading scripts
├── train.py               # Training entry point (Lightning-compatible)
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
This repo has been extensively reworked using Codex since I wanted to see if/how well it works. The AI changes wer concerned almost exclusively with structuring, commenting, and documentation. The GAN workflow itself was adapted from my previous implementations and the resulting experience with training these models: ([Remote-Sensing-SRGAN](https://github.com/simon-donike/Remote-Sensing-SRGAN)) and [NIR-GAN](https://github.com/simon-donike/NIR-GAN). The only exceptions are the loss class and the .SAFE dataset class, even though its AI slop it works for now so I won't touch them again.

## 🧑‍🚀 ToDOs  
- [ ] create inference.py  (interface with opensr-test)
- [ ] build interface with SEN2SR (for 10m + 20m SR)
- [x] incorporate the SEN2NAIP versions + downloading  
- [x] implement different discriminators
- [x] implement different visual loses (like LPIPS, VGG, ...)
- [ ] upgrade to torch>2.0 (complicated, PL doesnt support multiple schedulers in >2)
