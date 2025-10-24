# Data Pipeline

Components responsible for turning configuration files into ready-to-use PyTorch Lightning
data modules and applying reproducible normalization policies.

## Dataset selection

::: opensr_srgan.data.dataset_selector
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - select_dataset
        - datamodule_from_datasets
      filters:
        - "!^_"
      members_order: source

## Normalization utilities

::: opensr_srgan.data.utils.normalizer
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - Normalizer
      filters:
        - "!^_"
      members_order: source
