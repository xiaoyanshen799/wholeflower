"""ResNet-152 wrapper for CIFAR-100 with ImageNet-style input."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet152(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """Build torchvision ResNet-152 and replace the classifier head."""
    weights = None
    if pretrained:
        try:
            weights = models.ResNet152_Weights.IMAGENET1K_V2
        except AttributeError:
            try:
                weights = models.ResNet152_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = "IMAGENET1K_V1"

    model = models.resnet152(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
