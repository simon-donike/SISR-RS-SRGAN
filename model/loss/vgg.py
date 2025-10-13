"""VGG based perceptual feature extractor utilities."""

from torch import nn
import torchvision


class TruncatedVGG19(nn.Module):
    """A truncated VGG19 network used for perceptual loss computation."""

    def __init__(self, i: int = 5, j: int = 4) -> None:
        super().__init__()

        vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19.features.children():
            truncate_at += 1

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            if maxpool_counter == i - 1 and conv_counter == j:
                break

        if not (maxpool_counter == i - 1 and conv_counter == j):
            raise AssertionError(
                f"One or both of i={i} and j={j} are not valid choices for the VGG19!"
            )

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[: truncate_at + 1])

    def forward(self, input):  # type: ignore[override]
        """Forward propagation returning the requested VGG19 feature map."""
        return self.truncated_vgg19(input)


__all__ = ["TruncatedVGG19"]
