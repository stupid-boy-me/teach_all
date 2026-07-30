"""Image transforms for detection."""
from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: dict[str, Any]):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: dict[str, Any]):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: dict[str, Any]):
        if random.random() < self.p:
            _, _, width = image.shape
            image = image.flip(-1)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes = boxes.clone()
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                target["boxes"] = boxes
        return image, target


class ResizeShortSide:
    """Keep aspect ratio: short side -> short_side, long side <= max_size."""

    def __init__(self, short_side: int = 600, max_size: int = 1000):
        self.short_side = short_side
        self.max_size = max_size

    def __call__(self, image: torch.Tensor, target: dict[str, Any]):
        _, h, w = image.shape
        scale = self.short_side / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        image = F.resize(image, [nh, nw], antialias=True)
        if target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * scale
        # 一律用 Tensor，便于 engine 里 .to(device)
        target["scale"] = torch.tensor(scale, dtype=torch.float32)
        target["orig_size"] = torch.tensor([h, w], dtype=torch.int64)
        target["size"] = torch.tensor([nh, nw], dtype=torch.int64)
        return image, target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: dict[str, Any]):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def build_transforms(train: bool, short_side: int = 600, max_size: int = 1000) -> Compose:
    transforms = [ToTensor(), ResizeShortSide(short_side, max_size)]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(Normalize())
    return Compose(transforms)
