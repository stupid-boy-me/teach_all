"""
区域提案网络 RPN（结构三件套②）

本文件包含三个类：
  1. AnchorGenerator      —— 在特征图上生成预设 Anchor
  2. RPNHead              —— 3×3 + 双 1×1，预测 objectness 与框偏移
  3. RegionProposalNetwork—— 组装：出 proposals + 训练时算 RPN 两项损失

对应文档：docs/代码模块讲解/rpn区域提案网络.md
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from utils.boxes import (
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    remove_small_boxes,
    smooth_l1_loss,
)


class AnchorGenerator(nn.Module):
    """
    Anchor 生成器：在每个特征位置放置 k 个预设框（默认 3 尺度 × 3 比例 = 9）。

    作用：用「参考框金字塔」覆盖多尺度，从而不必对输入做图像金字塔。
    """

    def __init__(self, sizes=(64, 128, 256), aspect_ratios=(0.5, 1.0, 2.0), stride: int = 16):
        super().__init__()
        self.stride = stride  # 特征图 1 格对应原图像素数（与 Backbone stride 一致）
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(sizes) * len(aspect_ratios)
        # 以原点为中心的 k 个模板框，注册为 buffer（随模型移动到 GPU，但不作为可训练参数）
        self.register_buffer("cell_anchors", self._generate_cell_anchors(), persistent=False)

    def _generate_cell_anchors(self) -> torch.Tensor:
        """生成单个位置上的 k 个模板 Anchor，格式 xyxy，中心在 (0,0)。"""
        anchors = []
        for size in self.sizes:
            area = float(size) ** 2
            for ratio in self.aspect_ratios:
                # ratio = h/w；由面积反推宽高
                w = (area / ratio) ** 0.5
                h = w * ratio
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, feature: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """
        把模板框铺到整张特征图。

        返回: [H*W*k, 4] 的 xyxy Anchor（原图像素坐标）
        """
        _, _, fh, fw = feature.shape
        device = feature.device
        # 每个特征格子的中心映射回原图：(+0.5) * stride
        shift_x = (torch.arange(fw, device=device) + 0.5) * self.stride
        shift_y = (torch.arange(fh, device=device) + 0.5) * self.stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)
        # 每个中心点 + k 个模板 → 全部 Anchor
        anchors = (shifts[:, None, :] + self.cell_anchors[None, :, :]).reshape(-1, 4)
        return anchors


class RPNHead(nn.Module):
    """
    RPN 小网络头（论文里的滑窗网络）：
      共享 3×3 卷积 → 并行两个 1×1：
        - cls_logits: 每个位置 × k 个 objectness
        - bbox_pred:  每个位置 × k×4 个框偏移
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)  # → loss_objectness
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)  # → loss_rpn_box_reg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        t = F.relu(self.conv(x))
        logits = self.cls_logits(t)  # [B, k, H, W]
        bbox_reg = self.bbox_pred(t)  # [B, k*4, H, W]
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    """
    完整 RPN：
      前向始终：生成 proposals（训练/推理都要给后面 RoI 用，或仅 Step1/3 用）
      训练额外：匹配 GT → 采样 → 返回 loss_objectness + loss_rpn_box_reg
    """

    def __init__(
        self,
        in_channels: int,
        anchor_sizes=(64, 128, 256),
        anchor_ratios=(0.5, 1.0, 2.0),
        stride: int = 16,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_top_n_train: int = 2000,
        post_nms_top_n_train: int = 2000,
        pre_nms_top_n_test: int = 1000,
        post_nms_top_n_test: int = 300,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
    ):
        super().__init__()
        self.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios, stride)
        self.head = RPNHead(in_channels, self.anchor_generator.num_anchors)
        # 正负样本 IoU 阈值（论文设定）
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # NMS 前后保留数量：训练可多留，测试常用 300
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.post_nms_top_n_train = post_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

    def forward(
        self,
        features: torch.Tensor,
        images: list[torch.Tensor],
        targets: Optional[list[dict]] = None,
    ):
        """
        参数:
          features: Backbone 特征 [B,C,H,W]
          images:   原图列表（取高宽、算越界）
          targets:  训练时的 GT；推理传 None

        返回:
          proposals: list[Tensor[N,4]]
          losses:    训练时含 loss_objectness / loss_rpn_box_reg，否则 {}
        """
        logits, bbox_reg = self.head(features)
        b, a, h, w = logits.shape
        # 展平到「每个 Anchor 一个分数 / 一组 4 维偏移」
        objectness = logits.permute(0, 2, 3, 1).reshape(b, -1)
        pred_bbox_deltas = bbox_reg.permute(0, 2, 3, 1).reshape(b, -1, 4)

        proposals_per_image = []
        losses = {}
        all_anchors = []

        for i in range(b):
            img_h, img_w = images[i].shape[-2:]
            # 1) 生成该图全部 Anchor
            anchors = self.anchor_generator(features[i : i + 1], (img_h, img_w))
            all_anchors.append(anchors)
            # 2) Anchor + 预测偏移 → 解码成提案框，再裁剪到图像内
            proposals = decode_boxes(anchors, pred_bbox_deltas[i])
            proposals = clip_boxes_to_image(proposals, (img_h, img_w))
            keep = remove_small_boxes(proposals, min_size=1e-1)
            proposals = proposals[keep]
            scores = objectness[i].sigmoid()[keep]

            # 3) 高分 Top-K → NMS → 留下最终 proposals
            pre_nms = self.pre_nms_top_n_train if self.training else self.pre_nms_top_n_test
            post_nms = self.post_nms_top_n_train if self.training else self.post_nms_top_n_test
            topk = min(pre_nms, scores.numel())
            scores, idx = scores.topk(topk)
            proposals = proposals[idx]
            keep = nms(proposals, scores, self.nms_thresh)[:post_nms]
            proposals_per_image.append(proposals[keep])

        # 4) 训练时额外算 RPN 两项损失（正负样本在 _compute_loss 里匹配）
        if self.training and targets is not None:
            losses = self._compute_loss(objectness, pred_bbox_deltas, all_anchors, images, targets)

        return proposals_per_image, losses

    def _compute_loss(self, objectness, pred_bbox_deltas, all_anchors, images, targets):
        """
        匹配规则（论文）：
          IoU≥0.7 或对该 GT 最大 → 正样本；IoU<0.3 → 负样本；中间忽略。
        每图采样约 256 个（正负约 1:1）：
          - loss_objectness: BCE（分类）
          - loss_rpn_box_reg: 仅正样本 Smooth L1（相对 Anchor 回归）
        """
        labels_list = []
        regression_targets_list = []
        sampled_inds_list = []

        for i, (anchors, target) in enumerate(zip(all_anchors, targets)):
            gt_boxes = target["boxes"]
            img_h, img_w = images[i].shape[-2:]
            # 训练时忽略越界 Anchor（论文强调，否则难收敛）
            inside = (
                (anchors[:, 0] >= 0)
                & (anchors[:, 1] >= 0)
                & (anchors[:, 2] <= img_w)
                & (anchors[:, 3] <= img_h)
            )
            labels = torch.full((anchors.shape[0],), -1, dtype=torch.float32, device=anchors.device)
            if gt_boxes.numel() == 0:
                labels[inside] = 0
            else:
                ious = box_iou(anchors, gt_boxes)
                max_iou, matched_gt = ious.max(dim=1)
                labels[inside & (max_iou < self.bg_iou_thresh)] = 0
                labels[inside & (max_iou >= self.fg_iou_thresh)] = 1
                # 每个 GT 至少分配一个正样本（IoU 最大的那个 Anchor）
                best_anchor_per_gt = ious.argmax(dim=0)
                labels[best_anchor_per_gt] = 1
                matched_gt_boxes = gt_boxes[matched_gt]
                # 回归目标：GT 相对 Anchor 的编码
                regression_targets = encode_boxes(anchors, matched_gt_boxes)
                regression_targets_list.append(regression_targets)

            # 采样正负，控制比例
            pos = torch.where(labels == 1)[0]
            neg = torch.where(labels == 0)[0]
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(pos.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(neg.numel(), num_neg)
            perm_pos = torch.randperm(pos.numel(), device=anchors.device)[:num_pos]
            perm_neg = torch.randperm(neg.numel(), device=anchors.device)[:num_neg]
            sampled = torch.cat([pos[perm_pos], neg[perm_neg]], dim=0)

            labels_list.append(labels)
            sampled_inds_list.append(sampled)
            if gt_boxes.numel() == 0:
                regression_targets_list.append(torch.zeros_like(anchors))

        objectness_losses = []
        box_losses = []
        for i in range(len(images)):
            sampled = sampled_inds_list[i]
            labels = labels_list[i][sampled]
            obj = objectness[i][sampled]
            # 分类损失：物体 vs 背景
            objectness_losses.append(F.binary_cross_entropy_with_logits(obj, labels))

            pos_mask = labels > 0
            if pos_mask.any():
                pos_inds = sampled[pos_mask]
                # 回归损失：只对正样本 Anchor
                box_losses.append(
                    smooth_l1_loss(
                        pred_bbox_deltas[i][pos_inds],
                        regression_targets_list[i][pos_inds],
                    ).sum()
                    / max(pos_mask.sum(), 1)
                )
            else:
                box_losses.append(pred_bbox_deltas[i].sum() * 0.0)

        return {
            "loss_objectness": torch.stack(objectness_losses).mean(),
            "loss_rpn_box_reg": torch.stack(box_losses).mean(),
        }
