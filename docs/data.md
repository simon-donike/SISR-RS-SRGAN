# Data

ESA OpenSR provides dataset utilities tailored to remote-sensing super-resolution tasks. This guide summarises the available
options, required inputs, and how the Lightning data module is assembled.

## Dataset selector

`opensr_gan/data/data_utils.py` exposes `select_dataset(config)`, which inspects `config.Data.dataset_type` and returns a
`pytorch_lightning.LightningDataModule`. The helper keeps dataset instantiation and dataloader configuration in one place so that
training scripts only need the YAML file.

Supported dataset keys:

| Key | Description | Expected assets |
| --- | --- | --- |
| `S2_6b` | Sentinel-2 SAFE crops with six 20 m bands (B05, B06, B07, B8A, B11, B12). | Manifest JSON listing paired LR/HR chips, stored on disk with consistent band naming. |
| `S2_4b` | Sentinel-2 SAFE crops with four 10 m bands (B05, B04, B03, B02) reordered for RGB+NIR. | Same manifest structure as `S2_6b`; ensure 10 m bands are present. |
| `SISR_WW` | SEN2NAIP worldwide dataset. | Directory containing `train` and `val` splits compatible with the `SISRWorldWide` dataset class. |

Each dataset expects a super-resolution scale provided by `Generator.scaling_factor`. The helper passes this to the dataset class
as `sr_factor`, ensuring the dataloader and model agree on the upsampling ratio.

## Sentinel-2 SAFE datasets

Both Sentinel-2 options rely on `opensr_gan/data/SEN2_SAFE/S2_6b_ds.py` for data loading. Key behaviours include:

* **Manifest-driven tiling.** The JSON manifest encodes absolute paths to SAFE granule chips and ensures low-/high-resolution
  crops are spatially aligned.
* **Band selection and ordering.** `bands_keep` restricts which bands are loaded while `band_order` ensures the tensor channel
  ordering matches the model expectation.
* **On-the-fly antialiasing.** Downsampling uses antialiased transforms when `sr_factor > 1` to avoid aliasing artefacts during
  generator training.

When switching between `S2_6b` and `S2_4b`, verify that the manifest references the correct resolution tier (20 m vs 10 m) and
that the band names in the manifest align with the expected strings (`B05_20m`, `B04_10m`, etc.).

## SEN2NAIP worldwide dataset

`SISRWorldWide` provides global coverage pairs by fusing Sentinel-2 inputs with NAIP high-resolution targets. The dataset class
handles split logic internally; you only need to set the root path (defaults to `/data3/SEN2NAIP_global`). For best results:

* Keep the `train` and `val` directories consistent with the expected folder names.
* Confirm that the dynamic range of inputs matches the normalisation utilities used in the model (`normalise_10k`).

## Data module configuration

After instantiating the train and validation datasets, `select_dataset` wraps them in a minimal `LightningDataModule`:

* **Batch sizes.** Controlled via `Data.train_batch_size`, `Data.val_batch_size`, or the shared fallback `Data.batch_size`.
* **Workers and prefetching.** `Data.num_workers` and `Data.prefetch_factor` adjust dataloader throughput. When set to zero, the
  helper avoids passing `prefetch_factor` (which would otherwise raise an error).
* **Shuffling.** Training dataloaders shuffle by default, while validation loaders keep shuffling enabled for more diverse metric
  coverage across long epochs.
* **Logging.** A summary statement prints the dataset type and sample counts so you can confirm that manifests were parsed
  correctly.

Because the module returns a standard Lightning interface, you can drop it into custom training scripts or use it with
`Trainer.fit()` directly.

## Extending to new datasets

To integrate a new dataset:

1. Implement a `torch.utils.data.Dataset` that returns `(lr_img, hr_img)` tensors with channels in the expected order.
2. Add a new branch inside `select_dataset` that imports and instantiates your dataset class.
3. Update the documentation and your YAML configuration to expose the new `Data.dataset_type` key.

Following this pattern keeps your training scripts unchanged while enabling rapid experimentation with different sensors or tiling
strategies.
