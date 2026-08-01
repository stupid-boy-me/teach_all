# PyTorch 目标检测

> 分类是「这是猫」；检测是「猫在哪、有多大」。  
> 坐标一加，快乐翻倍，bug 也翻倍——欢迎来到视觉最热闹的赛道。

本目录从经典 2D 检测讲到多传感器 3D 融合，按项目独立环境/配置，**不要指望一个 `pip install` 打天下**。

---

## 项目一览

| 项目 | 一句话 | 适合谁 | 链接 |
|------|--------|--------|------|
| **YOLOv1** | 单阶段鼻祖，理解「网格 + 直接回归」 | 想搞懂 YOLO 家族祖先 | [yolov1](./yolov1) |
| **Faster R-CNN** | 两阶段代表：RPN + RoI Head，教学版联合训练 | 面试/论文对照/扎实基础 | [faster_rcnn](./faster_rcnn) |
| **BEVFusion** | Camera + LiDAR → BEV 3D 检测（MIT Han Lab） | 想碰自动驾驶感知 | [bevfusion_mit_han_lab](./bevfusion_mit_han_lab) |
| **格式转换** | VOC / YOLO 等标注格式互转小工具 | 数据清洗期救命 | [1.format_convert](./1.format_convert) |

---

## 推荐学习顺序

```text
1.format_convert（先别让标注格式坑你）
        ↓
     yolov1（建立单阶段直觉）
        ↓
  faster_rcnn（理解两阶段与四项 loss）
        ↓
bevfusion_mit_han_lab（多模态 + 3D，装备升级）
```

---

## 环境提醒（很重要）

| 项目 | 环境建议 |
|------|----------|
| YOLOv1 / 格式转换 | 可用仓库根 `requirement.txt` 或轻量自建环境 |
| Faster R-CNN | 见该目录 `requirements.txt` / README（conda `fasterrcnn`） |
| BEVFusion | **必须**按子目录 `install_env.sh`（独立 conda `bevfusion`） |

CUDA / PyTorch / mmcv 版本错配时，报错信息通常比你更有脾气——请先读对应 README。

---

## 常用数据

- VOC2007：见[根 README 网盘表](../README.md#数据集网盘常用)  
- nuScenes mini（BEVFusion）：见 [BEVFusion README](./bevfusion_mit_han_lab/README.md)

---

## 后续计划（占位）

YOLOv3 / YOLOv5 / YOLOX · RetinaNet · FCOS · DETR 系列 · 更多 3D 检测……  
路线图写在这里，是为了以后填坑时有坐标。

有问题加微信 `zuiaixiaopaomo`，附上命令和 traceback，我们一起把框画正。
