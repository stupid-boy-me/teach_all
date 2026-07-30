"""
Faster R-CNN 四步交替训练（论文 Section 3.2）——学习 / 复现用

仓库默认推荐联合训练：python train.py  或  python train_joint.py

本脚本流程：
Step1: 端到端训练 RPN（ImageNet 初始化 backbone）
Step2: 另建检测器，用 Step1 提案训练 Fast R-CNN（再次 ImageNet 初始化）
Step3: 用检测器 backbone 初始化，冻结共享卷积，只微调 RPN 独有层
Step4: 冻结共享卷积，只微调 Fast R-CNN 独有层

运行（在 faster_rcnn 目录下）:
  python download_voc.py
  python train_alternating.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import copy
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


def build_optimizer(model, cfg):
    params = trainable_params(model)
    if not params:
        raise RuntimeError("No trainable parameters — check freeze settings.")
    t = cfg["train"]
    return SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])


def run_step(
    name: str,
    model,
    data_loader,
    val_loader,
    device,
    cfg,
    epochs: int,
    proposal_model=None,
):
    print("\n" + "=" * 60)
    print(f" {name}")
    print("=" * 60)
    optimizer = build_optimizer(model, cfg)
    milestones = [e for e in cfg["train"]["lr_step_epochs"] if e < epochs]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=cfg["train"]["print_freq"],
            proposal_model=proposal_model,
        )
        scheduler.step()
        print(f"[{name}] epoch {epoch}/{epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        history.append(stats)

    eval_stats = evaluate_simple(
        model if model.mode == "full" else _as_full_eval(model, proposal_model),
        val_loader,
        device,
        score_thresh=cfg["eval"]["score_thresh"],
    )
    print(f"[{name}] quick-eval: {eval_stats}")
    return history


def _as_full_eval(model, proposal_model):
    """返回用于 quick-eval 的模型（不永久改写训练用 mode）。"""
    if model.mode == "full":
        return model
    wrapped = copy.deepcopy(model)
    if proposal_model is not None:
        wrapped.rpn.load_state_dict(proposal_model.rpn.state_dict())
    wrapped.mode = "full"
    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN 4-step alternating training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--download", action="store_true", help="自动下载 VOC")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = ROOT / cfg["train"]["output_dir"]
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

    # ------------------------------------------------------------------
    # Step 1: Train RPN end-to-end
    # ------------------------------------------------------------------
    model_rpn = build_model(cfg).to(device)
    model_rpn.mode = "rpn"
    model_rpn.set_trainable(backbone=True, rpn=True, roi_heads=False)
    run_step("Step1: Train RPN", model_rpn, train_loader, val_loader, device, cfg, cfg["train"]["step1_epochs"])
    save_checkpoint(
        {"model": model_rpn.state_dict(), "cfg": cfg, "step": 1},
        out_dir / "step1_rpn.pth",
    )

    # ------------------------------------------------------------------
    # Step 2: Train Fast R-CNN separately (fresh ImageNet backbone)
    # ------------------------------------------------------------------
    model_det = build_model(cfg).to(device)  # 重新 ImageNet 初始化
    model_det.mode = "rcnn"
    model_det.set_trainable(backbone=True, rpn=False, roi_heads=True)
    # Step1 RPN 冻结，只提供 proposals
    for p in model_rpn.parameters():
        p.requires_grad = False
    model_rpn.eval()
    run_step(
        "Step2: Train Fast R-CNN (fixed proposals from Step1)",
        model_det,
        train_loader,
        val_loader,
        device,
        cfg,
        cfg["train"]["step2_epochs"],
        proposal_model=model_rpn,
    )
    save_checkpoint(
        {"model": model_det.state_dict(), "cfg": cfg, "step": 2},
        out_dir / "step2_fast_rcnn.pth",
    )

    # ------------------------------------------------------------------
    # Step 3: Init from detector backbone, freeze shared conv, tune RPN head
    # ------------------------------------------------------------------
    model = build_model(cfg).to(device)
    # 共享卷积来自检测器；RPN 头可从 Step1 热启动
    model.backbone.load_state_dict(model_det.backbone.state_dict())
    model.roi_heads.load_state_dict(model_det.roi_heads.state_dict())
    model.rpn.load_state_dict(model_rpn.rpn.state_dict())
    model.mode = "rpn"
    model.set_trainable(backbone=False, rpn=True, roi_heads=False)
    run_step(
        "Step3: Fine-tune RPN unique layers (backbone frozen)",
        model,
        train_loader,
        val_loader,
        device,
        cfg,
        cfg["train"]["step3_epochs"],
    )
    save_checkpoint(
        {"model": model.state_dict(), "cfg": cfg, "step": 3},
        out_dir / "step3_rpn_ft.pth",
    )

    # ------------------------------------------------------------------
    # Step 4: Freeze shared conv, fine-tune Fast R-CNN unique layers
    # ------------------------------------------------------------------
    model.mode = "rcnn"
    model.set_trainable(backbone=False, rpn=False, roi_heads=True)
    # 用当前（已微调）RPN 提供提案
    proposal_ref = copy.deepcopy(model)
    for p in proposal_ref.parameters():
        p.requires_grad = False
    proposal_ref.eval()
    proposal_ref.mode = "rpn"
    run_step(
        "Step4: Fine-tune Fast R-CNN unique layers (backbone frozen)",
        model,
        train_loader,
        val_loader,
        device,
        cfg,
        cfg["train"]["step4_epochs"],
        proposal_model=proposal_ref,
    )

    model.mode = "full"
    model.set_trainable(backbone=True, rpn=True, roi_heads=True)  # 推理时全开
    save_checkpoint(
        {"model": model.state_dict(), "cfg": cfg, "step": 4},
        out_dir / "step4_final.pth",
    )
    print(f"\nDone. Final weights: {out_dir / 'step4_final.pth'}")
    print("下一步: python infer.py --checkpoint outputs/alternating/step4_final.pth --image /path/to.jpg")


if __name__ == "__main__":
    main()
