"""PASCAL VOC detection dataset wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VOCDetection

from utils.transforms import build_transforms

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 0 = background


def _parse_voc_target(target: dict) -> dict:
    objs = target["annotation"]["object"]
    if not isinstance(objs, list):
        objs = [objs]
    boxes = []
    labels = []
    for obj in objs:
        difficult = int(obj.get("difficult", 0))
        if difficult == 1:
            continue
        name = obj["name"]
        if name not in CLASS_TO_IDX:
            continue
        bnd = obj["bndbox"]
        box = [
            float(bnd["xmin"]),
            float(bnd["ymin"]),
            float(bnd["xmax"]),
            float(bnd["ymax"]),
        ]
        # VOC is 1-indexed inclusive; convert to xyxy float (keep as-is, common practice)
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append(box)
            labels.append(CLASS_TO_IDX[name])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "trainval",
        train: bool = True,
        short_side: int = 600,
        max_size: int = 1000,
        download: bool = False,
    ):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transforms = build_transforms(train=train, short_side=short_side, max_size=max_size)

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int):
        image, raw = self.voc[idx]
        target = _parse_voc_target(raw)
        target["image_id"] = torch.tensor([idx])
        image, target = self.transforms(image, target)
        return image, target


def build_voc_datasets(cfg: dict, download: bool = False):
    data = cfg["data"]
    root = data["root"]
    year = str(data["year"])
    train_ds = VOCDataset(
        root=root,
        year=year,
        image_set=data["train_split"],
        train=True,
        short_side=data["short_side"],
        max_size=data["max_size"],
        download=download,
    )
    val_ds = VOCDataset(
        root=root,
        year=year,
        image_set=data["val_split"],
        train=False,
        short_side=data["short_side"],
        max_size=data["max_size"],
        download=False,
    )
    subset = data.get("small_subset")
    if subset is not None:
        n = min(int(subset), len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        n_val = min(max(n // 4, 8), len(val_ds))
        val_ds = Subset(val_ds, list(range(n_val)))
    return train_ds, val_ds
