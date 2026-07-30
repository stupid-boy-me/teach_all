"""Bounding box utilities for Faster R-CNN."""
from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU matrix [N, M]. boxes in xyxy format."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return (ws >= min_size) & (hs >= min_size)


def encode_boxes(reference: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes relative to reference (anchors / proposals)."""
    wx = reference[:, 2] - reference[:, 0]
    wy = reference[:, 3] - reference[:, 1]
    cx = reference[:, 0] + 0.5 * wx
    cy = reference[:, 1] + 0.5 * wy

    px = proposals[:, 2] - proposals[:, 0]
    py = proposals[:, 3] - proposals[:, 1]
    pcx = proposals[:, 0] + 0.5 * px
    pcy = proposals[:, 1] + 0.5 * py

    eps = torch.finfo(reference.dtype).eps
    wx = wx.clamp(min=eps)
    wy = wy.clamp(min=eps)
    px = px.clamp(min=eps)
    py = py.clamp(min=eps)

    dx = (pcx - cx) / wx
    dy = (pcy - cy) / wy
    dw = torch.log(px / wx)
    dh = torch.log(py / wy)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(reference: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode box deltas relative to reference boxes."""
    wx = reference[:, 2] - reference[:, 0]
    wy = reference[:, 3] - reference[:, 1]
    cx = reference[:, 0] + 0.5 * wx
    cy = reference[:, 1] + 0.5 * wy

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2].clamp(max=4.0)  # avoid exp overflow
    dh = deltas[:, 3].clamp(max=4.0)

    pcx = dx * wx + cx
    pcy = dy * wy + cy
    pw = torch.exp(dw) * wx
    ph = torch.exp(dh) * wy

    x1 = pcx - 0.5 * pw
    y1 = pcy - 0.5 * ph
    x2 = pcx + 0.5 * pw
    y2 = pcy + 0.5 * ph
    return torch.stack((x1, y1, x2, y2), dim=1)


def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0 / 9) -> torch.Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss
