import torch
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


def configure_optimizers_PL1x(self):
    # configure Generator optimizer (Adam)
    optimizer_g = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, self.generator.parameters()),  # only trainable params
        lr=self.config.Optimizers.optim_g_lr                                     # LR from config
    )

    # configure Discriminator optimizer (Adam)
    optimizer_d = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),  # only trainable params
        lr=self.config.Optimizers.optim_d_lr                                       # LR from config
    )

    # learning rate schedulers (ReduceLROnPlateau)
    scheduler_g = ReduceLROnPlateau(
        optimizer_g, mode='min',
        factor=self.config.Schedulers.factor_g,
        patience=self.config.Schedulers.patience_g,
        #verbose=self.config.Schedulers.verbose
    )
    scheduler_d = ReduceLROnPlateau(
        optimizer_d, mode='min',
        factor=self.config.Schedulers.factor_d,
        patience=self.config.Schedulers.patience_d,
        #verbose=self.config.Schedulers.verbose
    )

    # optional generator warmup scheduler (step-based)
    warmup_steps = int(getattr(self.config.Schedulers, "g_warmup_steps", 0))
    warmup_type = getattr(self.config.Schedulers, "g_warmup_type", "none").lower()

    warmup_scheduler_config = None
    if warmup_steps > 0 and warmup_type in {"cosine", "linear"}:

        def _generator_warmup_lambda(current_step: int) -> float:
            if current_step >= warmup_steps:
                return 1.0

            progress = (current_step + 1) / float(max(1, warmup_steps))
            if warmup_type == "linear":
                return progress

            # default to cosine warmup for smoother start
            return 0.5 * (1.0 - math.cos(math.pi * progress))

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer_g,
            lr_lambda=_generator_warmup_lambda,
        )

        warmup_scheduler_config = {
            'scheduler': warmup_scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': 'generator_warmup',
        }

    scheduler_configs = [
        {'scheduler': scheduler_d, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1},
        {'scheduler': scheduler_g, 'monitor': self.config.Schedulers.metric, 'reduce_on_plateau': True, 'interval': 'epoch', 'frequency': 1}
    ]

    if warmup_scheduler_config is not None:
        scheduler_configs.append(warmup_scheduler_config)

    # return both optimizers + schedulers for PL
    return [
        [optimizer_d, optimizer_g],  # order super important, it's [D, G] and checked in training step
        scheduler_configs,
    ]
    
def configure_optimizers_PL2x(self):
    """Placeholder for PyTorch Lightning >= 2.0."""

    raise NotImplementedError(
        "PyTorch Lightning >= 2.0 support for configure_optimizers is not implemented yet."
    )