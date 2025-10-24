# Inference API

Utilities for exporting pretrained SRGAN checkpoints and running tiled inference on Sentinel-2
imagery.

## Core helpers

::: opensr_srgan.inference
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - load_model
        - run_sen2_inference
        - main
      filters:
        - "!^_"
      members_order: source

## Pretrained model factory

::: opensr_srgan._factory
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - load_from_config
        - load_inference_model
      filters:
        - "!^_"
      members_order: source
