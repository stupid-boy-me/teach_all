"""
Fast R-CNN 检测头 / RoI Head（结构三件套③）

本文件包含：
  1. FastRCNNPredictor —— 分类头 + 类相关框回归头
  2. RoIHeads          —— RoIAlign → FC → 预测；训练算损失，推理做 NMS

对应文档：docs/代码模块讲解/roi_head检测头.md
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, nms

from utils.boxes import box_iou, clip_boxes_to_image, decode_boxes, encode_boxes, smooth_l1_loss


class FastRCNNPredictor(nn.Module):
    """
    检测头最后的两路线性层：
      - cls_score: [N, num_classes]           → loss_classifier（多类 CE）
      - bbox_pred: [N, num_classes * 4]       → loss_box_reg（类相关回归）
    注意：回归是「每一类一套 4 维」，不是共用一套。
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor):
        return self.cls_score(x), self.bbox_pred(x)


class RoIHeads(nn.Module):
    """
    Fast R-CNN 头完整流程：
      训练：选正负 Proposal → RoIAlign → FC → 算分类/回归损失
      推理：RoIAlign → FC → 逐类 decode + NMS → 最终检测框
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        representation_size: int = 1024,
        roi_output_size: int = 7,
        spatial_scale: float = 1.0 / 16,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 128,
        positive_fraction: float = 0.25,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ):
        super().__init__()
        # 把每个 Proposal 从特征图抠成固定 7×7；spatial_scale 必须与 Backbone stride 对齐
        self.box_roi_pool = RoIAlign(
            output_size=roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        self.fc6 = nn.Linear(in_channels * roi_output_size * roi_output_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.box_predictor = FastRCNNPredictor(representation_size, num_classes)

        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        nn.init.kaiming_uniform_(self.fc6.weight, a=1)
        nn.init.kaiming_uniform_(self.fc7.weight, a=1)

    def forward(
        self,
        features: torch.Tensor,
        proposals: list[torch.Tensor],
        images: list[torch.Tensor],
        targets: Optional[list[dict]] = None,
    ):
        """
        参数:
          features:  共享 Backbone 特征（与 RPN 同一张）
          proposals: 候选框（来自本模型 RPN，或四步里外部老师 RPN）
          images:    原图（推理后处理裁剪用）
          targets:   训练 GT

        返回:
          result: 推理时的 detections 列表；训练时多为 []
          losses: 训练时含 loss_classifier / loss_box_reg
        """
        # 训练：先把 proposals 匹配成正负样本，并准备回归目标
        if self.training and targets is not None:
            proposals, matched_idxs, labels, regression_targets = self._select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # 每个框 → 固定尺寸特征 → 两层 FC → 分类 + 回归
        box_features = self.box_roi_pool(features, proposals)
        box_features = torch.flatten(box_features, start_dim=1)
        box_features = F.relu(self.fc6(box_features))
        box_features = F.relu(self.fc7(box_features))
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict] = []
        losses: dict = {}
        if self.training and labels is not None and regression_targets is not None:
            losses = self._compute_loss(class_logits, box_regression, labels, regression_targets)
        else:
            # 推理：softmax + 逐类 NMS，得到最终框
            boxes, scores, labels_out = self._postprocess_detections(
                class_logits, box_regression, proposals, images
            )
            for b, s, l in zip(boxes, scores, labels_out):
                result.append({"boxes": b, "scores": s, "labels": l})

        return result, losses

    def _select_training_samples(self, proposals, targets):
        """
        为 RoI 头准备训练样本：
          1) 把 GT 框拼进 proposals（保证有正样本）
          2) IoU 匹配：≥0.5 为正（该类），<0.5 为背景
          3) 采样约 128 个（正样本约 25%）
          4) 回归目标 = GT 相对 Proposal 的编码（注意：不是相对 Anchor）
        """
        sampled_props = []
        matched_idxs = []
        labels_all = []
        reg_targets_all = []

        for props, target in zip(proposals, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            # 经典 Fast R-CNN：训练时把 GT 也当作 proposal 喂进去
            props = torch.cat([props, gt_boxes], dim=0) if gt_boxes.numel() else props

            if gt_boxes.numel() == 0:
                labels = torch.zeros(props.shape[0], dtype=torch.int64, device=props.device)
                matched = torch.zeros(props.shape[0], dtype=torch.int64, device=props.device)
                reg_targets = torch.zeros(props.shape[0], 4, device=props.device)
            else:
                ious = box_iou(props, gt_boxes)
                max_iou, matched = ious.max(dim=1)
                labels = gt_labels[matched].clone()
                bg = max_iou < self.bg_iou_thresh
                labels[bg] = 0  # 背景类
                # fg/bg 阈值不同时，中间带标 -1 忽略（本配置默认两者都是 0.5）
                ignore = (max_iou >= self.bg_iou_thresh) & (max_iou < self.fg_iou_thresh)
                labels[ignore] = -1
                # 正样本 Proposal：相对 Proposal 编码 GT
                reg_targets = encode_boxes(props, gt_boxes[matched])

            pos = torch.where(labels > 0)[0]
            neg = torch.where(labels == 0)[0]
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(pos.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(neg.numel(), num_neg)
            pos = pos[torch.randperm(pos.numel(), device=props.device)[:num_pos]]
            neg = neg[torch.randperm(neg.numel(), device=props.device)[:num_neg]]
            keep = torch.cat([pos, neg], dim=0)

            sampled_props.append(props[keep])
            matched_idxs.append(matched[keep] if gt_boxes.numel() else matched)
            labels_all.append(labels[keep])
            reg_targets_all.append(reg_targets[keep])

        return sampled_props, matched_idxs, labels_all, reg_targets_all

    def _compute_loss(self, class_logits, box_regression, labels_list, regression_targets_list):
        """
        loss_classifier: 多类 CrossEntropy（分类损失）
        loss_box_reg:    仅正样本；取「真实类别」那一组 4 维做 Smooth L1（回归损失）
        """
        labels = torch.cat(labels_list, dim=0)
        regression_targets = torch.cat(regression_targets_list, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos]
        if sampled_pos.numel() > 0:
            # [N, num_classes*4] → [N, num_classes, 4]，再按真实类索引
            box_regression = box_regression.reshape(class_logits.shape[0], self.num_classes, 4)
            box_reg = box_regression[sampled_pos, labels_pos]
            box_loss = smooth_l1_loss(box_reg, regression_targets[sampled_pos]).sum() / max(
                labels.numel(), 1
            )
        else:
            box_loss = box_regression.sum() * 0.0

        return {"loss_classifier": classification_loss, "loss_box_reg": box_loss}

    def _postprocess_detections(self, class_logits, box_regression, proposals, images):
        """
        推理后处理：
          1) softmax 得各类分数
          2) 对每个前景类：用该类的 4 维偏移 decode Proposal
          3) 分数阈值过滤 + 该类内 NMS
          4) 各类合并后取 top detections_per_img
        """
        device = class_logits.device
        num_classes = self.num_classes
        start = 0
        boxes_out, scores_out, labels_out = [], [], []

        box_regression = box_regression.reshape(-1, num_classes, 4)
        pred_scores = F.softmax(class_logits, dim=-1)

        for props, image in zip(proposals, images):
            n = props.shape[0]
            deltas = box_regression[start : start + n]
            scores = pred_scores[start : start + n]
            start += n

            # 每个类各自 decode 一套框
            boxes = decode_boxes(
                props.unsqueeze(1).expand(-1, num_classes, -1).reshape(-1, 4),
                deltas.reshape(-1, 4),
            ).reshape(n, num_classes, 4)
            img_h, img_w = image.shape[-2:]
            boxes = clip_boxes_to_image(boxes.reshape(-1, 4), (img_h, img_w)).reshape(
                n, num_classes, 4
            )

            # 丢掉背景类 0
            boxes = boxes[:, 1:, :]
            scores = scores[:, 1:]

            boxes_per_img, scores_per_img, labels_per_img = [], [], []
            for cls_id in range(num_classes - 1):
                score = scores[:, cls_id]
                keep = score > self.score_thresh
                box = boxes[keep, cls_id, :]
                score = score[keep]
                if score.numel() == 0:
                    continue
                keep_idx = nms(box, score, self.nms_thresh)
                box = box[keep_idx]
                score = score[keep_idx]
                label = torch.full((score.numel(),), cls_id + 1, dtype=torch.int64, device=device)
                boxes_per_img.append(box)
                scores_per_img.append(score)
                labels_per_img.append(label)

            if boxes_per_img:
                boxes_cat = torch.cat(boxes_per_img, dim=0)
                scores_cat = torch.cat(scores_per_img, dim=0)
                labels_cat = torch.cat(labels_per_img, dim=0)
                topk = min(self.detections_per_img, scores_cat.numel())
                scores_cat, idx = scores_cat.topk(topk)
                boxes_out.append(boxes_cat[idx])
                scores_out.append(scores_cat)
                labels_out.append(labels_cat[idx])
            else:
                boxes_out.append(torch.zeros((0, 4), device=device))
                scores_out.append(torch.zeros((0,), device=device))
                labels_out.append(torch.zeros((0,), dtype=torch.int64, device=device))

        return boxes_out, scores_out, labels_out
