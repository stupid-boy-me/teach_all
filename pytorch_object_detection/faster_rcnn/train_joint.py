"""
Faster R-CNN 默认训练入口：近似联合训练（推荐）

每个 iter：
  mode=full → ① Backbone → ② RPN → ③ RoI
  总损失 = loss_objectness + loss_rpn_box_reg + loss_classifier + loss_box_reg
  一次 backward，①②③ 一起更新（无 Step1~4 切换）

若要复现论文四步交替，请用：python train_alternating.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.voc import build_voc_datasets
from engine import evaluate_simple, train_one_epoch
from models.faster_rcnn import build_model
from utils.misc import collate_fn, load_config, save_checkpoint, set_seed, trainable_params


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN 联合训练（推荐默认）")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 joint_epochs")
    parser.add_argument("--download", action="store_true", help="自动下载 VOC")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("训练策略: 近似联合训练 (mode=full, 无四步切换)")

    epochs = args.epochs if args.epochs is not None else int(cfg["train"].get("joint_epochs", 12))
    out_dir = ROOT / cfg["train"].get("joint_output_dir", "./outputs/joint")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_voc_datasets(cfg, download=args.download)
    print(f"Train images: {len(train_ds)}  Val images: {len(val_ds)}")

    loader_kw = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    # ★ 联合训练核心：一条龙 + 三件套全部可训练
    model.mode = "full"
    model.set_trainable(backbone=True, rpn=True, roi_heads=True)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    optimizer = SGD(
        trainable_params(model),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    # 联合训练：在总 epoch 的 2/3 处降 lr（也可用配置 joint_lr_step_epochs）
    milestones = cfg["train"].get("joint_lr_step_epochs")
    if milestones is None:
        milestones = [max(epochs * 2 // 3, 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_det_ratio = -1.0
    for epoch in range(start_epoch, epochs + 1):
        stats = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            print_freq=cfg["train"]["print_freq"],
            proposal_model=None,  # 联合训练：自己的 RPN 出框，无需外部老师
        )
        scheduler.step()
        print(
            f"[Joint] epoch {epoch}/{epochs}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        )

        eval_stats = evaluate_simple(
            model, val_loader, device, score_thresh=cfg["eval"]["score_thresh"]
        )
        print(f"[Joint] eval: {eval_stats}")
        if eval_stats["det_image_ratio"] >= best_det_ratio:
            best_det_ratio = eval_stats["det_image_ratio"]
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "strategy": "joint",
                    "eval": eval_stats,
                },
                out_dir / "joint_best.pth",
            )

    final_path = out_dir / "joint_final.pth"
    save_checkpoint(
        {"model": model.state_dict(), "cfg": cfg, "epoch": epochs, "strategy": "joint"},
        final_path,
    )
    print(f"\nDone. 推荐推理权重: {final_path}")
    print("或使用验证更好的: ", out_dir / "joint_best.pth")
    print(
        f"python infer.py --checkpoint {final_path.as_posix()} --image /path/to.jpg"
    )


if __name__ == "__main__":
    main()
