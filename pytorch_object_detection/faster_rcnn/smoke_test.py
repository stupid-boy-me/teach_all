"""无 VOC 也可跑的冒烟测试：构建模型 + 假数据前向。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.faster_rcnn import build_model
from utils.misc import load_config


def main():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["model"]["pretrained_backbone"] = False  # 离线可跑
    device = torch.device("cpu")
    model = build_model(cfg).to(device)

    img = torch.rand(3, 400, 600)
    target = {
        "boxes": torch.tensor([[50.0, 60.0, 200.0, 300.0]]),
        "labels": torch.tensor([3], dtype=torch.int64),
    }

    # RPN train
    model.train()
    model.mode = "rpn"
    model.set_trainable(True, True, False)
    losses = model([img], [target])
    print("Step1-like RPN losses:", {k: float(v) for k, v in losses.items()})
    (sum(losses.values())).backward()
    print("RPN backward OK")

    # RCNN train with external proposals
    model.zero_grad()
    model.mode = "rcnn"
    model.set_trainable(True, False, True)
    props = [torch.tensor([[40.0, 50.0, 220.0, 320.0], [10.0, 10.0, 80.0, 90.0]])]
    losses = model([img], [target], proposals=props)
    print("Step2-like RCNN losses:", {k: float(v) for k, v in losses.items()})
    (sum(losses.values())).backward()
    print("RCNN backward OK")

    # Full inference
    model.eval()
    model.mode = "full"
    outs = model([img])
    print("Infer boxes:", outs[0]["boxes"].shape)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
