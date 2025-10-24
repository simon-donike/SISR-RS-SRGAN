# API Reference

The API reference documents the Python entry points that back the configuration-driven
super-resolution workflows. Each page is rendered with
[mkdocstrings](https://mkdocstrings.github.io/), ensuring the signatures and docstrings stay in
sync with the implementation.

## Package overview

::: opensr_srgan
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - SRGAN_model
        - train
        - load_from_config
        - load_inference_model
