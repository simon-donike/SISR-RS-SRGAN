# Training Workflow

Remote-Sensing-SRGAN uses PyTorch Lightning to manage training loops, logging, and checkpointing. This guide explains the runtime workflow, callbacks, and practical tips for stable training on large remote-sensing datasets.

## End-to-end flow

1. **Configuration load:** `train.py` reads the YAML file supplied via `--config` and passes it to `SRGAN_model`.
2. **Dataset selection:** `select_dataset` constructs train/validation datasets and wraps them into a Lightning `DataModule`. Dataset statistics are printed for sanity.
3. **Logger setup:** Weights & Biases, TensorBoard, and a learning-rate monitor are initialised. Checkpoints are saved under `logs/<project>/<timestamp>/`.
4. **Training loop:** `Trainer.fit` launches GPU-accelerated training with callbacks for checkpointing, early stopping, and LR monitoring.

```python
--8<-- "../train.py:lines=19-113"
```

```python
--8<-- "../data/data_utils.py:lines=1-150"
```

## Lightning hooks inside `SRGAN_model`

* `training_step`: Receives both generator and discriminator optimisers (Lightning’s GAN pattern). Handles generator-only pretraining, adversarial ramp-up, and logging of scalar losses/metrics.
* `validation_step`: Generates super-resolved outputs for qualitative logging, computes metrics (PSNR, SSIM, LPIPS if configured), and stores them for epoch-level aggregation.
* `configure_optimizers`: Creates two Adam optimisers plus `ReduceLROnPlateau` schedulers based on config values.
* `on_validation_epoch_end`: Uses `utils.logging_helpers.plot_tensors` to push LR/SR/HR panels to TensorBoard and Weights & Biases.

```python
--8<-- "../model/SRGAN.py:lines=195-443"
```

```python
--8<-- "../utils/logging_helpers.py:lines=1-72"
```

Inspect `model/SRGAN.py` for detailed comments describing each step of the training loop and logging behaviour.

## Callbacks and logging

`train.py` wires in several Lightning callbacks out of the box:

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Saves `last.ckpt` and top-2 checkpoints based on `Schedulers.metric` (default: `val_metrics/l1`).|
| `LearningRateMonitor` | Logs generator/discriminator learning rates each epoch to W&B/TensorBoard.|
| `EarlyStopping` | Monitors the same validation metric with large patience (250 epochs) to guard against divergence.|

```python
--8<-- "../train.py:lines=59-110"
```

Weights & Biases is configured with `project="SRGAN_6bands"` and entity `opensr`. Set the `WANDB_PROJECT` or edit the script if you need per-experiment projects. TensorBoard logs are stored in `logs/` alongside W&B run data.

## Adversarial training schedule

GAN training is stabilised through three mechanisms:

* **Generator pretraining:** For the first `Training.g_pretrain_steps`, only the generator optimiser runs; the discriminator is skipped. This lets the generator learn a strong content prior before adversarial updates start.
* **Adversarial ramp-up:** After pretraining, the adversarial loss weight increases linearly or sigmoidally over `Training.adv_loss_ramp_steps` until it reaches `Losses.adv_loss_beta`.
* **Label smoothing:** When `Training.label_smoothing=True`, real labels are reduced to 0.9 to prevent discriminator overconfidence.

```python
--8<-- "../model/SRGAN.py:lines=34-58"
```

```yaml
--8<-- "../configs/config_10m.yaml:lines=35-70"
```

These heuristics reduce mode collapse and stabilise training on multi-spectral inputs where illumination and texture vary drastically between bands.

## Practical tips

* Monitor PSNR/SSIM/LPIPS in W&B to detect spectral artefacts early. If LPIPS diverges while PSNR improves, consider increasing `Losses.perceptual_weight`.
* Use moderate `train_batch_size` values (8–16) to balance GPU utilisation and dataset variety. Increase `Data.prefetch_factor` when `num_workers > 0` to keep GPUs busy.

```python
--8<-- "../data/data_utils.py:lines=97-150"
```
* To benchmark architectures quickly, reuse trained generators by setting `Model.load_checkpoint` and adjusting only the discriminator or loss weights.
* Keep an eye on GPU memory when scaling `Generator.n_blocks` and `n_channels`; RRDB and LKA variants are heavier than SRResNet.

