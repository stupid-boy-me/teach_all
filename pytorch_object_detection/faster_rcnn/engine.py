"""Training / evaluation loops."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.misc import AverageMeter


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
    proposal_model: Optional[torch.nn.Module] = None,
) -> dict[str, float]:
    """
    proposal_model: 若提供，则用其（eval）生成固定 proposals，供 Step2/Step4 的 rcnn 模式使用。
    """
    model.train()
    if proposal_model is not None:
        proposal_model.eval()

    meters = {k: AverageMeter() for k in ("loss", "loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg")}
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)

    for step, (images, targets) in enumerate(pbar):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

        proposals = None
        if proposal_model is not None:
            with torch.no_grad():
                # 用冻结的 RPN 模型生成提案
                old_mode = proposal_model.mode
                proposal_model.mode = "rpn"
                feats = []
                for img in images:
                    feats.append(proposal_model.backbone(img.unsqueeze(0)))
                features = torch.cat(feats, dim=0)
                proposals, _ = proposal_model.rpn(features, images, targets=None)
                proposals = [p.detach() for p in proposals]
                proposal_model.mode = old_mode

        loss_dict = model(images, targets, proposals=proposals)
        losses = sum(loss_dict.values())

        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss: {loss_dict}")

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        meters["loss"].update(losses.item(), n=len(images))
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v.item(), n=len(images))

        if step % print_freq == 0:
            pbar.set_postfix({k: f"{m.avg:.3f}" for k, m in meters.items() if m.count > 0})

    return {k: m.avg for k, m in meters.items() if m.count > 0}


@torch.no_grad()
def evaluate_simple(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.5,
) -> dict[str, float]:
    """轻量检查：统计有检测框的图片比例与平均框数（完整 mAP 见 evaluate.py）。"""
    model.eval()
    old_mode = model.mode
    model.mode = "full"
    n_img = 0
    n_with_det = 0
    n_boxes = 0
    for images, targets in tqdm(data_loader, desc="Eval", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out in outputs:
            n_img += 1
            keep = out["scores"] >= score_thresh
            nb = int(keep.sum().item())
            n_boxes += nb
            if nb > 0:
                n_with_det += 1
    model.mode = old_mode
    return {
        "images": float(n_img),
        "det_image_ratio": n_with_det / max(n_img, 1),
        "avg_boxes": n_boxes / max(n_img, 1),
    }
