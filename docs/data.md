# Data Pipelines

Remote-Sensing-SRGAN ships with dataset loaders tailored to Sentinel-2 SAFE products and the SEN2NAIP worldwide corpus. All loaders produce `(lr, hr)` tensor pairs suitable for Lightning training and plug into the config-driven selector in `data/data_utils.py`.

## Dataset selector

`data.data_utils.select_dataset(config)` reads `config.Data.dataset_type` and constructs the appropriate dataset instances. The helper then wraps them into a minimal Lightning `DataModule` with configurable batch sizes, worker counts, and prefetch factors.

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "1-150"}
```

```python
pl_datamodule = select_dataset(config)
train_loader = pl_datamodule.train_dataloader()
val_loader = pl_datamodule.val_dataloader()
```

The selector currently supports three dataset families:

### Sentinel-2 SAFE 6-band (`S2_6b`)

* **Goal:** Train on six 20 m Sentinel-2 bands (B05, B06, B07, B8A, B11, B12) stacked into a single tensor.
* **Implementation:** Uses `data/SEN2_SAFE/S2_6b_ds.py::S2SAFEDataset` to read a manifest of pre-windowed chips. The dataset handles normalisation, band ordering, LR synthesis via anti-aliased downsampling, and invalid chip filtering.

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "17-48"}
```
* **Configuration hooks:** `Generator.scaling_factor` determines the SR scale (2×/4×/8×). Hard-coded manifest path and band order can be promoted to config keys if you need to vary them frequently.

### Sentinel-2 SAFE 4-band (`S2_4b`)

* **Goal:** Focus on RGB+NIR super-resolution using four 10 m bands (B05, B04, B03, B02).
* **Implementation:** Reuses `S2SAFEDataset` with an alternative band list. Despite the 10 m intent, the manifest path currently points to the 20 m JSON; adjust it if you curate a dedicated 10 m manifest.

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "50-82"}
```
* **Configuration hooks:** Same as `S2_6b`; update band order and manifest in the dataset branch if you require different inputs.

### SEN2NAIP Worldwide (`SISR_WW`)

* **Goal:** Leverage Taco Foundation’s SEN2NAIP pairs where Sentinel-2 observations are paired with NAIP aerial imagery at 4× resolution.
* **Implementation:** The `SISRWorldWide` dataset (under `data/SISR_WW/`) reads the dataset root, honours `split` arguments (`train`/`val`), and returns aligned `(lr, hr)` samples. The data selector simply instantiates train/val splits pointing to `/data3/SEN2NAIP_global` by default.

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "84-95"}
```
* **Configuration hooks:** Update the root path or expose it through the config to point at your local dataset copy.

## Building new datasets

Adding a dataset follows three simple steps:

1. **Implement the dataset class** under `data/<name>/` with `__len__` and `__getitem__` returning `(lr, hr)` arrays or tensors.
2. **Register the selector** by adding a new `elif` branch in `select_dataset` that instantiates your dataset and passes the config values you need.
3. **Expose configuration keys** under `Data` (e.g., `manifest_path`, `bands_keep`) so experiments remain reproducible without code edits.

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "1-95"}
```

## DataLoader configuration

`datamodule_from_datasets` handles DataLoader construction and mirrors config values to PyTorch Lightning:

* `train_batch_size` / `val_batch_size`: separate sizes for each split, falling back to `Data.batch_size` if provided.
* `num_workers`: enables multi-process loading; when set to zero the helper disables persistent workers and prefetch factors.
* `prefetch_factor`: forwarded when workers > 0 to balance host-device throughput.
* `shuffle`: enabled for both train and validation to improve spatial diversity in evaluation batches (intentional design choice).

```python
--8<-- "data/data_utils.py"
--8<-- {"lines": "97-150"}
```

Because the `DataModule` is assembled dynamically, you can plug in custom datasets without changing the training loop — just add a branch and update your YAML.

