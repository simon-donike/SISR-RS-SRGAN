# Model Components

Classes and building blocks used to assemble SRGAN training and inference graphs.

## Lightning module

::: opensr_srgan.model.SRGAN
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - SRGAN_model
      filters:
        - "!^_"
      members_order: source

## Building blocks

::: opensr_srgan.model.model_blocks
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members:
        - ConvolutionalBlock
        - SubPixelConvolutionalBlock
        - ResidualBlock
        - ResidualBlockNoBN
        - RCAB
        - DenseBlock5
        - RRDB
        - LKA
        - LKAResBlock
        - make_upsampler
        - ExponentialMovingAverage
      members_order: source

## Generator architectures

::: opensr_srgan.model.generators
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members_order: source

## Discriminator architectures

::: opensr_srgan.model.discriminators
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members_order: source

## Loss components

::: opensr_srgan.model.loss.loss
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - GeneratorContentLoss
      filters:
        - "!^_"
      members_order: source

::: opensr_srgan.model.loss.vgg
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      members:
        - TruncatedVGG19
      filters:
        - "!^_"
      members_order: source

## Training step helpers

::: opensr_srgan.model.training_step_PL
    handler: python
    options:
      show_root_heading: false
      show_root_full_path: false
      show_source: true
      filters:
        - "!^_"
      members_order: source
