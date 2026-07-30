"""Shared convolutional backbone (ResNet C4-style feature)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class ResNetBackbone(nn.Module):
    """
    使用 ResNet 的 layer1~layer3 输出特征图（约 stride=16），
    对应论文中与 RPN / Fast R-CNN 共享的卷积层。
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=weights)
            self.out_channels = 256
        elif name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.out_channels = 1024
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.stride = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
