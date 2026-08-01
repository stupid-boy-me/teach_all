# BEVFusion 算法结构说明（3D 目标检测）

本文说明本仓库（MIT Han Lab BEVFusion）如何做多传感器 3D 检测，代码落在哪里，以及你该跑哪条配置。

论文：[BEVFusion (ICRA 2023)](https://arxiv.org/abs/2205.13542)  
官网：[bevfusion.mit.edu](http://bevfusion.mit.edu/)

---

## 1. 核心思想（用一句话说清）

传统做法常把相机特征投影到 LiDAR 点上（点级融合），容易丢掉相机的语义密度。

BEVFusion 换了一条路：

1. 相机 → 提特征 → **视角变换到 BEV**
2. LiDAR → 体素化 → **稀疏卷积得到 BEV**
3. 在 **同一鸟瞰平面** 上融合
4. 再用检测头输出 3D 框

这样几何（LiDAR）和语义（相机）都能保留，且检测 / BEV 分割可共用同一套融合骨架。

```text
  6×相机图像 ──► Camera Backbone/Neck ──► View Transform ──► Camera BEV ─┐
                                                                          ├─► Fuser ─► BEV Decoder ─► 3D Detection Head
  LiDAR 点云 ──► Voxelize ──► Sparse Encoder ──────────────► LiDAR BEV ─┘
```

---

## 2. 仓库代码地图

```text
bevfusion_mit_han_lab/
├── configs/nuscenes/          # 训练/评测 YAML（模态、网络、数据增强）
├── mmdet3d/
│   ├── datasets/              # NuScenes 数据集与 pipeline
│   ├── models/
│   │   ├── fusion_models/bevfusion.py   # ★ 总模型
│   │   ├── backbones/                   # Swin / SparseEncoder / SECOND ...
│   │   ├── necks/                       # LSS FPN / SECOND FPN ...
│   │   ├── vtransforms/                 # 相机→BEV（LSS / DepthLSS ...）
│   │   ├── fusers/                      # ConvFuser / AddFuser
│   │   └── heads/bbox/                  # TransFusion / CenterPoint 检测头
│   └── ops/                   # CUDA 算子（bev_pool、spconv、voxel 等）
└── tools/
    ├── create_data.py         # 生成 infos / GT database
    ├── train.py               # 训练入口
    ├── test.py                # 评测入口
    └── visualize.py           # 可视化
```

总模型类：`mmdet3d/models/fusion_models/bevfusion.py` 中的 `BEVFusion`。

构造时四大块：

| 模块 | 含义 |
|------|------|
| `encoders` | 相机分支 / LiDAR 分支（可选雷达） |
| `fuser` | BEV 特征融合（纯单模态时可为空） |
| `decoder` | 融合后的 BEV 骨干 + FPN |
| `heads` | `object` 检测头 和/或 `map` 分割头 |

---

## 3. 各模块在做什么

### 3.1 相机分支（Camera Encoder）

典型融合检测配置路径：

`configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/`

流程：

1. **Backbone**：`SwinTransformer`（也可换成 ResNet 等）
2. **Neck**：`GeneralizedLSSFPN`，输出统一通道的多尺度特征
3. **View Transform**：`DepthLSSTransform`（Lift-Splat-Shoot 思想）
   - 估计每个像素的深度分布
   - 把图像特征「抬」到 3D，再「泼」到 BEV 栅格
   - 本仓库对 BEV pooling 做了高效 CUDA 实现（`mmdet3d/ops/bev_pool`）

输出：相机 BEV 特征图（融合配置里通道常为 80）。

### 3.2 LiDAR 分支（LiDAR Encoder）

1. **Voxelize**：把点云划进体素（`voxel_size` 如 `0.075×0.075×0.2`）
2. **SparseEncoder**：3D 稀疏卷积，得到鸟瞰特征

输出：LiDAR BEV 特征图（通道常为 256）。

### 3.3 融合器（Fuser）

`ConvFuser`（`mmdet3d/models/fusers/conv.py`）：

- 把相机 BEV 与 LiDAR BEV **在通道维拼接**
- 再做 `Conv + BN + ReLU`

配置示例（`convfuser.yaml`）：

```yaml
model:
  fuser:
    type: ConvFuser
    in_channels: [80, 256]   # camera, lidar
    out_channels: 256
```

也有简单的 `AddFuser`（逐元素相加）。

### 3.4 BEV Decoder

通常：

- Backbone：`SECOND`
- Neck：`SECONDFPN`

把融合后的 BEV 进一步编码，送给检测头。

### 3.5 检测头（Object Head）

融合检测默认用 **TransFusion** 头（`mmdet3d/models/heads/bbox/transfusion.py`）：

- 在 BEV 上查询目标
- 回归 3D 框：中心、尺寸、旋转、速度等
- 分类 10 类（见数据集文档）

相机-only 基线常用 CenterPoint 风格头；LiDAR-only 基线也是 TransFusion-L。

---

## 4. 三种常用模态（怎么选）

| 配置 | 模态 | 说明 | 相对难度 |
|------|------|------|----------|
| `.../lidar/voxelnet_0p075.yaml` | 仅 LiDAR | 几何强、流程相对简单 | ★★ |
| `.../camera/.../swint/default.yaml` | 仅相机 | 依赖视角变换，精度通常低于 LiDAR | ★★★ |
| `.../camera+lidar/.../convfuser.yaml` | 相机+LiDAR | 完整 BEVFusion，精度最高 | ★★★★ |

**建议你的路径（RTX 3060 12GB + mini）：**

1. 先跑 **LiDAR-only** 打通数据与训练
2. 再跑 **Camera+LiDAR** 完整融合
3. 需要刷指标时再换完整 nuScenes trainval

官方验证集参考（完整数据，非 mini）：

| 模型 | mAP | NDS |
|------|-----|-----|
| Camera-only | 35.56 | 41.21 |
| LiDAR-only | 64.68 | 69.28 |
| BEVFusion (C+L) | 68.52 | 71.38 |

---

## 5. 配置是如何叠起来的

本仓库用 **torchpack** 递归合并同目录下的 `default.yaml`。例如融合检测：

```text
configs/default.yaml
configs/nuscenes/default.yaml
configs/nuscenes/det/default.yaml
configs/nuscenes/det/transfusion/default.yaml
configs/nuscenes/det/transfusion/secfpn/default.yaml
configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml   ← 你指定的入口
```

你改 batch size、类别、学习率时，优先改最具体的那一层，或训练时用命令行覆盖。

---

## 6. 前向数据流（对应代码）

`BEVFusion.forward_single` 大致步骤：

1. 若有相机：`extract_camera_features(...)` → camera BEV  
2. 若有 LiDAR：`extract_features(points, "lidar")` → lidar BEV  
3. `fuser([camera_bev, lidar_bev])`（单模态则跳过）  
4. `decoder.backbone` → `decoder.neck`  
5. `heads["object"]` 算 loss（训练）或解码框（推理）

训练入口：`tools/train.py`  
评测入口：`tools/test.py`（检测用 `--eval bbox`）

---

## 7. 与「只做车辆检测」的关系

- 算法本身是 **通用 3D 目标检测**，默认 10 类（含行人、锥桶等）。
- 「车辆检测」不是换算法，而是：
  1. 先用官方 10 类跑通；
  2. 再在配置 `object_classes` 中只保留车辆类，并同步改 head / 评估相关设置。

mini 上不要期待接近论文的 mAP；目标是验证整条链路。

---

## 8. 显存与算力提示（RTX 3060 12GB）

完整融合模型较重：

- 官方默认 `samples_per_gpu: 4`（8 卡场景），单卡 12GB 很容易 OOM。
- 建议起步：`samples_per_gpu: 1`，`workers_per_gpu: 2`。
- 可先用 LiDAR-only + mini 验证。
- 预训练权重：`./tools/download_pretrained.sh`（评测/微调用）。

---

## 9. 环境就绪后的最小命令清单

> 先按回复中的环境章节建好 `bevfusion` conda 环境并 `python setup.py develop`。

```bash
cd /workspace/project/codes/bevfusion_mit_han_lab

# 1) 数据软链接
mkdir -p data
ln -sfn /workspace/datasets/v1.0-mini data/nuscenes

# 2) 生成 infos
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --max-sweeps 10

# 3) （可选）下载预训练
./tools/download_pretrained.sh

# 4) 单卡训练示例：LiDAR-only
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml \
  --data.samples_per_gpu 1

# 5) 单卡训练示例：完整 BEVFusion
torchpack dist-run -np 1 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
  --load_from pretrained/lidar-only-det.pth \
  --data.samples_per_gpu 1
```

评测示例：

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox
```

---

## 10. 推荐阅读顺序（结合代码）

1. `configs/nuscenes/default.yaml` — 数据字段与 10 类  
2. `mmdet3d/models/fusion_models/bevfusion.py` — 总前向  
3. `mmdet3d/models/vtransforms/` — 相机如何变成 BEV  
4. `mmdet3d/models/fusers/conv.py` — 融合如何发生  
5. `mmdet3d/models/heads/bbox/transfusion.py` — 框怎么出来  

读完以上五处，整条 3D 检测链路就串起来了。
