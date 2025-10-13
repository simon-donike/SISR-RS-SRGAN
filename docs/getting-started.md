# Getting Started

This guide walks through environment setup, configuration management, and the minimal commands required to launch a Remote-Sensing-SRGAN experiment.

## Prerequisites

* Python 3.10 (required by the pinned PyTorch 1.13 / torchvision 0.14 wheels).【F:README.md†L45-L73】
* CUDA-capable GPU if you plan to train on large Sentinel-2 chips (the default trainer is configured for GPU execution).【F:train.py†L69-L90】
* Optional: Weights & Biases account for experiment tracking and Git LFS if you plan to download large datasets.

## Create an environment

```bash
# Clone the repository
git clone https://github.com/ESAOpenSR/Remote-Sensing-SRGAN.git
cd Remote-Sensing-SRGAN

# (optional) Create a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# (recommended) Upgrade pip
python -m pip install --upgrade pip

# Install runtime dependencies
pip install -r requirements.txt
```

If the main PyPI index cannot resolve the exact PyTorch build, install the wheel directly from the official index before `pip install -r requirements.txt`:

```bash
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
```

## Configure your experiment

All experiment settings live in YAML files under `configs/`. The `config_20m.yaml` template targets Sentinel-2 20 m inputs, while `config_10m.yaml` demonstrates 10 m multi-band training. Key sections include:

* `Data`: batch sizes, worker counts, and dataset selector (`S2_6b`, `S2_4b`, `SISR_WW`, etc.).【F:configs/config_10m.yaml†L12-L33】
* `Model`: input band count and checkpoint loading behaviour.【F:configs/config_10m.yaml†L22-L28】
* `Training`: warm-up lengths, adversarial ramp strategy, and loss weights.【F:configs/config_10m.yaml†L35-L70】
* `Generator` / `Discriminator`: architecture choices and scale factor.【F:configs/config_10m.yaml†L73-L101】
* `Optimizers`, `Schedulers`, `Logging`: learning rates, ReduceLROnPlateau settings, and validation visualisations.【F:configs/config_10m.yaml†L103-L132】

Duplicate a config file if you need to adjust parameters without touching the defaults:

```bash
cp configs/config_20m.yaml configs/my_experiment.yaml
```

## Launch training

Use the training script with the desired YAML file:

```bash
python train.py --config configs/my_experiment.yaml
```

Under the hood the script:

1. Loads the configuration via OmegaConf and instantiates the Lightning `SRGAN_model`.【F:train.py†L24-L48】
2. Selects and wraps the dataset into a Lightning `DataModule` based on `Data.dataset_type`.【F:train.py†L49-L57】【F:data/data_utils.py†L1-L95】
3. Configures Weights & Biases, TensorBoard, and learning rate monitoring callbacks. 【F:train.py†L59-L93】
4. Starts training with GPU acceleration enabled by default. 【F:train.py†L69-L90】

Training logs, checkpoints, and exported validation panels are written into `logs/` alongside the W&B run.

## Resume or fine-tune

Two checkpoint switches let you reuse trained weights without editing Python code:

* `Model.load_checkpoint`: path to a Lightning checkpoint whose weights should initialise the generator/discriminator before training begins. 【F:train.py†L34-L47】
* `Model.continue_training`: path to a checkpoint that should be fully resumed (optimizer states, schedulers, etc.). Leave both as `False` to start from scratch.【F:train.py†L34-L47】

## Inference quick peek

For offline inference, instantiate `SRGAN_model` and call `predict_step` with low-resolution tensors. The method auto-normalises Sentinel-2 style inputs, runs the generator, histogram matches the output, and denormalises back to the original range.【F:model/SRGAN.py†L103-L144】

```python
from model.SRGAN import SRGAN_model
model = SRGAN_model(config_file_path="configs/my_experiment.yaml")
model.eval()

with torch.no_grad():
    sr = model.predict_step(lr_tensor)
```

