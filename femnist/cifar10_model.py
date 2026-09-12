"""CIFAR-10 ResNet-18 model."""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        stem_padding: int = 1,
        use_maxpool: bool = False,
        avgpool_kernel_size: int = 4,
        avgpool_stride: int = 4,
    ):
        super().__init__()
        self.in_planes = base_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            padding=stem_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if use_maxpool
            else nn.Identity()
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_kernel_size, stride=avgpool_stride)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(block(self.in_planes, planes, current_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    """ResNet-18 with CIFAR-10-friendly stem defaults."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        stem_padding: int = 1,
        use_maxpool: bool = False,
        avgpool_kernel_size: int = 4,
        avgpool_stride: int = 4,
    ):
        super().__init__()
        self.model = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            stem_kernel_size=stem_kernel_size,
            stem_stride=stem_stride,
            stem_padding=stem_padding,
            use_maxpool=use_maxpool,
            avgpool_kernel_size=avgpool_kernel_size,
            avgpool_stride=avgpool_stride,
        )
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    return ResNet18(
        in_channels=3,
        num_classes=num_classes,
        base_channels=64,
        stem_kernel_size=3,
        stem_stride=1,
        stem_padding=1,
        use_maxpool=False,
        avgpool_kernel_size=4,
        avgpool_stride=4,
    )
