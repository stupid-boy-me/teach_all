"""
Faster R-CNN 整网组装（把三件套①②③装在一起）

本文件：
  - FasterRCNN: Backbone + RPN + RoIHeads
  - mode / set_trainable: 支持四步交替与联合训练
  - build_model: 从配置构建

对应文档：docs/代码模块讲解/faster_rcnn整网组装.md
说明：backbone.py 本身较直观，模块讲解文档中不再单独展开。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.backbone import ResNetBackbone
from models.roi_head import RoIHeads
from models.rpn import RegionProposalNetwork
from utils.misc import freeze_module


class FasterRCNN(nn.Module):
    """
    Faster R-CNN = ①共享 Backbone + ②RPN + ③RoI Head

    mode 控制「这次前向通电到哪」：
      - 'rpn'  : 只跑 ①→②（Step1 / Step3）
      - 'rcnn' : 只跑 ①→③，proposals 从外部喂入（Step2 / Step4）
      - 'full' : ①→②→③ 一条龙（联合训练 / 推理）

    set_trainable 控制「通电的块里谁允许更新参数」（四步冻层用）。
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        # ① 共享卷积：只算一次特征，后面 RPN 与 RoI 共用
        self.backbone = ResNetBackbone(m["backbone"], m["pretrained_backbone"])
        in_channels = self.backbone.out_channels
        stride = self.backbone.stride

        # ② 区域提案网络
        self.rpn = RegionProposalNetwork(
            in_channels=in_channels,
            anchor_sizes=tuple(m["anchor_sizes"]),
            anchor_ratios=tuple(m["anchor_ratios"]),
            stride=stride,
            fg_iou_thresh=m["rpn_fg_iou"],
            bg_iou_thresh=m["rpn_bg_iou"],
            batch_size_per_image=m["rpn_batch_size"],
            positive_fraction=m["rpn_positive_fraction"],
            pre_nms_top_n_train=m["rpn_pre_nms_top_n_train"],
            post_nms_top_n_train=m["rpn_post_nms_top_n_train"],
            pre_nms_top_n_test=m["rpn_pre_nms_top_n_test"],
            post_nms_top_n_test=m["rpn_post_nms_top_n_test"],
            nms_thresh=m["rpn_nms_thresh"],
        )
        # ③ 检测头（Fast R-CNN）；spatial_scale=1/stride 与特征对齐
        self.roi_heads = RoIHeads(
            in_channels=in_channels,
            num_classes=m["num_classes"],
            spatial_scale=1.0 / stride,
            fg_iou_thresh=m["box_fg_iou"],
            bg_iou_thresh=m["box_bg_iou"],
            batch_size_per_image=m["box_batch_size"],
            positive_fraction=m["box_positive_fraction"],
            score_thresh=m["box_score_thresh"],
            nms_thresh=m["box_nms_thresh"],
            detections_per_img=m["box_detections_per_img"],
        )
        self.mode = "full"

    def set_trainable(self, backbone: bool, rpn: bool, roi_heads: bool) -> None:
        """四步交替时冻层：True=可训练，False=冻结。"""
        freeze_module(self.backbone, freeze=not backbone)
        freeze_module(self.rpn, freeze=not rpn)
        freeze_module(self.roi_heads, freeze=not roi_heads)

    def forward(
        self,
        images: list[torch.Tensor],
        targets: Optional[list[dict]] = None,
        proposals: Optional[list[torch.Tensor]] = None,
    ):
        """
        参数:
          images:    已归一化的 [C,H,W] 列表
          targets:   训练 GT：boxes(xyxy) + labels
          proposals: 可选；Step2/4 由外部老师 RPN 传入

        返回:
          训练 → losses 字典
          推理 → detections 列表（boxes / scores / labels）
        """
        # —— ① Backbone：提共享特征 ——
        # batch 内尺寸相同则一次 stack；否则逐图前向（默认 batch_size=1）
        image_tensors = (
            torch.stack([img for img in images], dim=0) if self._same_size(images) else None
        )

        if image_tensors is None:
            features_list = [self.backbone(img.unsqueeze(0)) for img in images]
            features = torch.cat(features_list, dim=0)
        else:
            features = self.backbone(image_tensors)

        losses = {}
        detections: list[dict] = []

        # —— ② RPN：mode 为 rpn / full 时运行 ——
        if self.mode in ("rpn", "full"):
            # 训练传 targets 以算 RPN 损失；推理传 None
            props, rpn_losses = self.rpn(features, images, targets if self.training else None)
            losses.update(rpn_losses)
            if proposals is None:
                proposals = props  # 自产 proposals；rcnn 模式则用外部传入的

        assert proposals is not None, "proposals required in rcnn mode"

        # —— ③ RoI Head：mode 为 rcnn / full 时运行 ——
        if self.mode in ("rcnn", "full"):
            # 近似联合 / Step2：框坐标当常数，不把梯度回传到「怎么回归出框」
            if self.mode == "rcnn" or (self.mode == "full" and self.training):
                proposals = [p.detach() for p in proposals]
            dets, roi_losses = self.roi_heads(
                features, proposals, images, targets if self.training else None
            )
            losses.update(roi_losses)
            detections = dets

        # 训练只需要 loss；推理只需要检测结果
        if self.training:
            return losses
        return detections

    @staticmethod
    def _same_size(images: list[torch.Tensor]) -> bool:
        h, w = images[0].shape[-2:]
        return all(img.shape[-2:] == (h, w) for img in images)


def build_model(cfg: dict) -> FasterRCNN:
    """从 configs/default.yaml 一类配置构建模型。"""
    return FasterRCNN(cfg)
