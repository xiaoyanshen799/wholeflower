"""EfficientNet model wrapper for ISIC2019."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_efficientnet(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with a custom classifier head."""
    weights = None
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = "IMAGENET1K_V1"

    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
