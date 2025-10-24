# Utility Helpers

Reusable helpers for logging, radiometric preprocessing, tensor conversion, and distributed-safe
side effects.

## Package exports

::: opensr_srgan.utils
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members_order: source

## Logging helpers

::: opensr_srgan.utils.logging_helpers
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Radiometric transforms

::: opensr_srgan.utils.radiometrics
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Tensor conversions

::: opensr_srgan.utils.tensor_conversions
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members_order: source

## Model summarisation

::: opensr_srgan.utils.model_descriptions
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - print_model_summary
      members_order: source

## Distributed coordination

::: opensr_srgan.utils.gpu_rank
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^__"
      members:
        - _is_global_zero
      members_order: source
