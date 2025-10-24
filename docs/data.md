# Data

The training stack ships with a single, self-contained example dataset that you can download in seconds to verify that the pipeline works end to end. This page explains how to fetch the sample data, how it is structured, and what you need to do when
you are ready to plug in your own collections.

## Example dataset

The example dataset is a small Sentinel-2 crop bundle hosted on the Hugging Face Hub. Each `.npz` archive contains a high-resolution chip stored under the key `hr`. Low-resolution counterparts are generated on the fly by bicubic interpolation inside the dataset class.

* **Scale factor:** 4× upsampling between the generated LR inputs and provided HR targets.
* **Splits:** All files except the final 20 samples are used for training; the last 20 form the validation split.
* **Normalisation:** Values above 1.5 are assumed to be Sentinel-2 reflectance and are normalised by `1/10000`.

### Downloading the files

`opensr_srgan/data/example_data/download_example_dataset.py` exposes a helper that downloads and extracts the archive into `example_dataset/` relative to your working directory. Run it from a Python shell or a small script:

```python
from opensr_srgan.data.example_data.download_example_dataset import get_example_dataset

get_example_dataset()
```

The helper pulls `example_dataset.zip` from the [`simon-donike/SR-GAN`](https://huggingface.co/simon-donike/SR-GAN) repository, extracts it, and removes the temporary archive once the copy completes.

### Directory layout

After extraction the folder contains `.npz` chips named `hr_*.npz`. No further preparation is required. Simply point the configuration to the dataset by setting:

```yaml
Data:
  dataset_type: ExampleDataset
```

The training loop automatically instantiates `opensr_srgan.data.example_data.example_dataset.ExampleDataset` for both the
training and validation dataloaders.

## Adding new datasets

When you are ready to move beyond the bundled sample data you can register a custom dataset. The repository uses a single factory function `opensr_srgan.data.dataset_selector.select_dataset` to keep the training script agnostic of individual
collections. To integrate a new source:

1. **Implement a dataset class.** Create a `torch.utils.data.Dataset` that returns `(lr, hr)` tensors and performs any normalisation your sensor requires. Place the implementation somewhere under `opensr_srgan/data/`.
2. **Update the selector.** Add a new `elif` branch to `select_dataset` that imports your dataset class, instantiates the training and validation splits, and returns them.
3. **Expose configuration hooks.** Introduce a new `Data.dataset_type` key (for example `MyDataset`) and any additional fields you need (paths, augmentation flags, scale factors, …). Document the required entries in your configuration file.

Following this pattern keeps `opensr_srgan.train` untouched: once the selector knows about your dataset you can launch training through the CLI or the Python API without further changes.
