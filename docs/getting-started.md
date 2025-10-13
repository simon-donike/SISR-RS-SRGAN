# Getting Started

This guide walks through environment setup, configuration management, and the minimal commands required to launch a Remote-Sensing-SRGAN experiment.

## Prerequisites

The project pins PyTorch 1.13 and torchvision 0.14 wheels that target Python 3.10. Set up your environment with that interpreter and install dependencies before launching training.

```markdown
--8<-- "README.md:lines=52-83"
```

The default trainer is configured for CUDA acceleration and streams metrics to both TensorBoard and Weights & Biases.

```python
--8<-- "train.py:lines=69-110"
```

Create a Weights & Biases account if you plan to use the built-in experiment tracking, and install Git LFS when working with large dataset manifests.

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

All experiment settings live in YAML files under `configs/`. The `config_20m.yaml` template targets Sentinel-2 20 m inputs, while `config_10m.yaml` demonstrates 10 m multi-band training. Each section controls loaders, models, training schedules, and logging behaviour.

```yaml
--8<-- "configs/config_10m.yaml:lines=12-132"
```

Duplicate a config file if you need to adjust parameters without touching the defaults:

```bash
cp configs/config_20m.yaml configs/my_experiment.yaml
```

## Launch training

Use the training script with the desired YAML file:

```bash
python train.py --config configs/my_experiment.yaml
```

Under the hood the script loads the YAML, constructs datasets, wires loggers and callbacks, and launches GPU-accelerated training with Lightning.

```python
--8<-- "train.py:lines=19-113"
```

Training logs, checkpoints, and exported validation panels are written into `logs/` alongside the W&B run.

## Resume or fine-tune

Two checkpoint switches let you reuse trained weights without editing Python code:

```python
--8<-- "train.py:lines=34-48"
```

## Inference quick peek

For offline inference, instantiate `SRGAN_model` and call `predict_step` with low-resolution tensors. The method auto-normalises Sentinel-2 style inputs, runs the generator, histogram matches the output, and denormalises back to the original range.

```python
--8<-- "model/SRGAN.py:lines=153-192"
```

```python
from model.SRGAN import SRGAN_model
model = SRGAN_model(config_file_path="configs/my_experiment.yaml")
model.eval()

with torch.no_grad():
    sr = model.predict_step(lr_tensor)
```

