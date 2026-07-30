"""简单评估入口（快速熟悉用；完整 COCO/VOC mAP 可后续接入）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.voc import build_voc_datasets
from engine import evaluate_simple
from models.faster_rcnn import build_model
from utils.misc import collate_fn, load_checkpoint, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    _, val_ds = build_voc_datasets(cfg, download=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt["model"])
    model.mode = "full"
    print(evaluate_simple(model, val_loader, device, cfg["eval"]["score_thresh"]))


if __name__ == "__main__":
    main()
