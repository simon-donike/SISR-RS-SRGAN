# Training workflow

`train.py` is the canonical entry point for ESA OpenSR experiments. It ties together configuration loading, model instantiation,
dataset selection, logging, and callbacks. This page explains how the script is organised and how to customise the training loop.

## Command-line interface

Run the script with a single optional argument:

```bash
python train.py --config path/to/config.yaml
```

* `--config / -c`: Path to a YAML file describing the experiment. Defaults to `configs/config_20m.yaml`.

The script sets `CUDA_VISIBLE_DEVICES="0"` by default. Override this environment variable before launching if you want to select
a different GPU or enable multi-GPU training.

## Initialisation steps

1. **Import dependencies.** Torch, PyTorch Lightning, OmegaConf, and logging backends are loaded up-front.
2. **Parse arguments.** `argparse` reads the configuration path and ensures the file exists.
3. **Load configuration.** `OmegaConf.load()` parses the YAML file into an object used throughout the run.
4. **Construct the model.**
   * If `Model.load_checkpoint` is set, the script calls `SRGAN_model.load_from_checkpoint()` to reuse learned weights while
     respecting the new configuration values.
    * Otherwise, it initialises a fresh `SRGAN_model`, which immediately builds the generator/discriminator and prints a
      parameter summary.
5. **Resume training (optional).** If `Model.continue_training` points to a Lightning checkpoint, the resulting path is passed to
   `Trainer(..., resume_from_checkpoint=...)` so optimiser states and schedulers resume where they left off.

## Data module construction

`select_dataset(config)` decides which dataset pair to use and wraps them in a `LightningDataModule`. The module inherits batch
sizes, worker counts, and prefetching parameters from the configuration and prints a summary including dataset size.

## Logging setup

* **Weights & Biases.** `WandbLogger` records scalar metrics, adversarial diagnostics, and validation image panels.
* **TensorBoard.** `TensorBoardLogger` writes the same scalar metrics locally under `logs/<project>/<timestamp>`.
* **Manual SummaryWriter.** A temporary TensorBoard writer (`logs/tmp`) remains available for quick custom logging if needed.

To disable W&B logging, either remove the logger from the list or unset your API key before launching the script.

## Callbacks

The following callbacks are registered with the Lightning trainer:

| Callback | Purpose |
| --- | --- |
| `ModelCheckpoint` | Saves the top two checkpoints according to `Schedulers.metric` and always keeps the last epoch. |
| `LearningRateMonitor` | Logs learning rates for both optimisers every epoch. |
| `EarlyStopping` | Monitors the same metric as the schedulers with a patience of 250 epochs and finite-check enabled. |

Checkpoint directories are nested under the TensorBoard log folder using the W&B project name and a timestamp, making it easy to
correlate files across tooling.

## Trainer configuration

The script builds a `Trainer` with the following notable arguments:

* `accelerator='cuda'` and `devices=[0]` for single-GPU runs. Adjust to `devices=[0,1,...]` or use `devices='auto'` for multi-GPU
  training.
* `check_val_every_n_epoch=1` to evaluate after every epoch.
* `limit_val_batches=250` as a safeguard against excessive validation time on large datasets.
* `logger=[wandb_logger]` to register external logging backends (add `tb_logger` if you prefer TensorBoard-driven monitoring).
* `callbacks=[checkpoint_callback, early_stop_callback, lr_monitor]` to activate the components described above.

Finally, `trainer.fit(model, datamodule=pl_datamodule)` launches the optimisation loop and `wandb.finish()` ensures clean shutdown
of the W&B session.

## Generator EMA lifecycle

If `Training.EMA.enabled` is `True`, the Lightning module keeps a shadow copy of the generator weights using the decay set in
`Training.EMA.decay`. The EMA state:

* updates immediately after each generator optimiser step once `Training.EMA.update_after_step` has been reached,
* lives on the device requested via `Training.EMA.device` (falling back to the generator's device), and
* automatically swaps in for evaluation, testing, and inference before being restored for continued training.

Checkpoints store both the live and EMA weights, so resuming training preserves the smoothed model.

## Practical tips

* **Gradient stability.** Tune `Training.pretrain_g_only`, `g_pretrain_steps`, and `adv_loss_ramp_steps` when experimenting with
  new generator architectures. Longer warm-ups often help deeper networks converge.
* **Learning-rate warmup.** `Schedulers.g_warmup_steps` and `Schedulers.g_warmup_type` apply a step-wise warmup (cosine or linear)
  to the generator LR before handing control back to the plateau scheduler. Start with 1–5k steps to avoid shocking freshly
  initialised weights.
* **Checkpoint hygiene.** Periodically prune the timestamped checkpoint directories to reclaim disk space, especially after
  exploratory runs.
* **Validation images.** Reduce `Logging.num_val_images` if logging slows down training, or set it to zero to disable qualitative
  logging entirely.
* **Experiment tracking.** Use descriptive W&B run names by exporting `WANDB_NAME="S2_8x_rrdb"` before launching the script.
* **EMA tuning.** Adjust `Training.EMA.decay` between 0.995 and 0.9999 depending on how aggressively you want to smooth the
  generator. Lower values react faster but may track noise; higher values provide the cleanest validation swaps.

With these components understood, you can safely modify the trainer arguments, replace callbacks, or integrate advanced logging
without losing the benefits of the existing automation.
