"""单图推理可视化。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.voc import VOC_CLASSES
from models.faster_rcnn import build_model
from utils.misc import load_checkpoint, load_config
from utils.transforms import Normalize, ResizeShortSide


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="outputs/pred.jpg")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else args.checkpoint)
    model.load_state_dict(ckpt["model"])
    model.mode = "full"
    model.eval()

    pil = Image.open(args.image).convert("RGB")
    img = TF.to_tensor(pil)
    img, meta = ResizeShortSide(cfg["data"]["short_side"], cfg["data"]["max_size"])(img, {"boxes": torch.zeros((0, 4))})
    img, _ = Normalize()(img, meta)
    outputs = model([img.to(device)])[0]

    draw = ImageDraw.Draw(pil)
    scale = meta.get("scale", 1.0)
    keep = outputs["scores"] >= args.score_thresh
    boxes = outputs["boxes"][keep].cpu()
    scores = outputs["scores"][keep].cpu()
    labels = outputs["labels"][keep].cpu()

    # 映射回原图尺度
    boxes = boxes / float(scale)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        name = VOC_CLASSES[int(label) - 1] if 1 <= int(label) <= 20 else str(int(label))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{name} {score:.2f}", fill="yellow")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    print(f"Saved visualization to {out_path}  (detections={keep.sum().item()})")


if __name__ == "__main__":
    main()
