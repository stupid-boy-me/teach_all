# BEVFusion 可复现实验（nuScenes mini）

基于 [MIT Han Lab BEVFusion](https://github.com/mit-han-lab/bevfusion) 的 **Camera + LiDAR 3D 目标检测** 实践仓库。  
本仓库重点不是复述论文，而是提供：**一键环境安装 → mini 数据准备 → 训练 / 评测 / 可视化** 的完整可跑通流程，并记录在 WSL2 + 较新 NVIDIA 驱动下踩过的坑与修复。

| 项目 | 说明 |
|------|------|
| 上游代码 | [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) |
| 论文 | [BEVFusion (ICRA 2023)](https://arxiv.org/abs/2205.13542) |
| 官方站点 | [bevfusion.mit.edu](http://bevfusion.mit.edu/) |
| 官方完整 README | 请直接阅读上游仓库：[mit-han-lab/bevfusion/README.md](https://github.com/mit-han-lab/bevfusion/blob/master/README.md) |

> **重要**：下文实验指标均在 **nuScenes mini** 上测得，**不能**与官方 nuScenes full val / test 榜单数字直接对比。

---

## 目录

1. [本仓库做了什么](#1-本仓库做了什么)
2. [本机已跑通结果](#2-本机已跑通结果)
3. [环境要求](#3-环境要求)
4. [快速开始](#4-快速开始)
5. [预训练权重说明](#5-预训练权重说明)
6. [训练 / 评测 / 可视化](#6-训练--评测--可视化)
7. [目录结构](#7-目录结构)
8. [常见问题与已知限制](#8-常见问题与已知限制)
9. [文档](#9-文档)
10. [上游项目与引用](#10-上游项目与引用)

---

## 1. 本仓库做了什么

相对官方仓库，本仓库额外提供 / 修补了：

- **一键安装**：`install_env.sh`（conda 环境、依赖钉扎、CUDA 扩展编译、自动下载官方 `pretrained/`）
- **一键运行**：`run_bevfusion.sh`（demo / train-lidar / train-fusion / eval）
- **数据**：面向 **nuScenes mini** 的软链接 + `create_data` 流程说明
- **兼容性修复**（在 PyTorch 1.10 + 系统 CUDA 12.x / 新驱动上常见）：
  - NumPy：`np.bool` / `np.long` 等旧别名
  - `setuptools` / `yapf` / `tensorboard` 与 mmcv、torch 的版本冲突
  - DepthLSS 深度通道与 `get_cam_feats(..., mats_dict)` 签名
  - 稀疏卷积 batch=1 / 空体素导致的 BN 与 CUDA error 9
  - `tools/visualize.py`：`tqdm` 导入、失败样本跳过、`--max-samples`
- **中文说明**：`docs/` 下数据集与算法结构笔记

---

## 2. 本机已跑通结果

**硬件 / 环境参考**

- GPU：NVIDIA GeForce RTX 3060 12GB
- 系统：WSL2 / Docker 类环境，主机驱动较新（CUDA 驱动侧 12.x / 13.x）
- 训练框架：Python 3.8 + PyTorch 1.10.1 + cudatoolkit 11.3 + mmcv-full 1.4.0

**任务**：Camera + LiDAR 融合 3D 检测（`convfuser.yaml`）  
**数据**：nuScenes **mini**（约 train 323 / val 81，经 CBGS 后每 epoch 约 815 iter，`BATCH=2`）  
**训练**：6 epochs，加载官方 `swint-nuimages-pretrained.pth` + `lidar-only-det.pth`

| Epoch | NDS | mAP | car AP@2.0m |
|------:|----:|----:|------------:|
| 1 | 0.332 | 0.310 | 0.856 |
| 2 | 0.363 | 0.293 | 0.828 |
| 3 | 0.370 | 0.304 | 0.844 |
| 4 | 0.334 | 0.302 | 0.865 |
| 5 | 0.392 | 0.324 | 0.861 |
| **6** | **0.400** | **0.328** | **0.864** |

- 训练日志示例：`runs/run-945d8599/20260801_103127.log`（本地）
- 权重示例：`runs/run-945d8599/epoch_6.pth`（本地；**默认不进 Git**，体积大）
- 可视化示例：本地运行后生成 `viz/epoch6/`（相机 6 路 + LiDAR 俯视图）

---

## 3. 环境要求

| 项目 | 建议版本 |
|------|----------|
| Python | 3.8 |
| PyTorch | 1.10.1 + cudatoolkit 11.3 |
| mmcv-full | 1.4.0（需匹配 cu113 / torch1.10 的 wheel） |
| mmdet | 2.20.0 |
| OpenMPI / mpi4py | conda-forge（torchpack 分布式启动用） |
| GPU | 显存建议 ≥ 8GB；融合 `BATCH=2` 在 12GB 上约 9GB |

更细的包列表见仓库内 `env.txt`（环境导出记录）。

---

## 4. 快速开始

### 4.1 克隆代码

若从本教学仓库使用：

```bash
git clone https://github.com/stupid-boy-me/teach_all.git
cd teach_all/pytorch_object_detection/bevfusion_mit_han_lab
```

### 4.2 一键安装环境 + 官方预训练

```bash
bash install_env.sh
```

常用选项：

```bash
bash install_env.sh --force              # 强制重建 conda 环境
bash install_env.sh --skip-compile       # 跳过 CUDA 扩展编译
bash install_env.sh --with-data          # 额外做数据软链接 + create_data
# 数据根目录可用环境变量覆盖，例如：
NUSCENES_SRC=/path/to/v1.0-mini bash install_env.sh --with-data
```

安装完成后：

```bash
conda activate bevfusion
```

`install_env.sh` 会在缺少权重时自动执行 `tools/download_pretrained.sh`，将官方 checkpoint 下到 `pretrained/`（约 865MB）。

### 4.3 准备 nuScenes mini 数据

1. 从 [nuScenes](https://www.nuscenes.org/download) 下载 **v1.0-mini**
2. 软链接到本仓库期望路径（或改 `NUSCENES_SRC`）：

```bash
mkdir -p data
ln -sfn /你的路径/v1.0-mini data/nuscenes
```

期望能看到类似结构：

```text
data/nuscenes/
  ├── maps/
  ├── samples/
  ├── sweeps/
  ├── v1.0-mini/
  └── ...
```

3. 生成 infos / GT database（若未用 `--with-data`）：

```bash
conda activate bevfusion
python tools/create_data.py nuscenes \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --max-sweeps 10
```

成功后应出现例如：

- `data/nuscenes/nuscenes_infos_train.pkl`
- `data/nuscenes/nuscenes_infos_val.pkl`
- `data/nuscenes/nuscenes_dbinfos_train.pkl`
- `data/nuscenes/nuscenes_gt_database/`

数据集字段与样本含义见：[`docs/01_nuScenes_数据集说明.md`](docs/01_nuScenes_数据集说明.md)。

### 4.4 一键跑通（推荐顺序）

```bash
conda activate bevfusion

# 1) LiDAR-only 快速推理（最稳，验证链路）
bash run_bevfusion.sh demo 10

# 2) 融合训练（需 pretrained 中的 swint + lidar-only）
EPOCHS=6 BATCH=2 bash run_bevfusion.sh train-fusion

# 3) 可视化自己的 checkpoint（路径按实际 runs/ 修改）
torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --mode pred \
  --checkpoint runs/run-XXXX/epoch_6.pth \
  --split val \
  --bbox-score 0.3 \
  --out-dir viz/epoch6
```

环境变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `EPOCHS` | 6 | 训练轮数 |
| `BATCH` | 2 | `samples_per_gpu`，建议 ≥ 2（更稳）；OOM 可改 `1` |

---

## 5. 预训练权重说明

权重**默认不进入 Git**（体积大）。通过 `bash tools/download_pretrained.sh` 或安装脚本自动下载到 `pretrained/`。

| 文件 | 约大小 | 是否必需 | 用途 |
|------|--------|----------|------|
| `swint-nuimages-pretrained.pth` | 106MB | **融合训练必需** | Camera Swin-T 初始化 |
| `lidar-only-det.pth` | 32MB | **融合训练必需** | `--load_from`；LiDAR demo/评测 |
| `bevfusion-det.pth` | 157MB | 评测/对比用 | 官方融合检测模型 |
| `camera-only-det.pth` | 170MB | 可选 | 相机检测 baseline |
| `*-seg.pth`（多个） | — | 分割任务才需要 | BEV map segmentation |

本仓库当前一键脚本主路径是 **检测（det）**；分割请参考[官方 README](https://github.com/mit-han-lab/bevfusion/blob/master/README.md)。

`mmdet3d.egg-info/` 是 `python setup.py develop` 生成的安装元数据，**无需上传、无需拷贝**。

---

## 6. 训练 / 评测 / 可视化

### 6.1 `run_bevfusion.sh` 子命令

```bash
bash run_bevfusion.sh demo [N]      # LiDAR-only 推理前 N 帧
bash run_bevfusion.sh train-lidar   # 训练 LiDAR-only
bash run_bevfusion.sh train-fusion  # 训练 Camera+LiDAR
bash run_bevfusion.sh eval-lidar    # 评测官方 lidar-only-det.pth
bash run_bevfusion.sh eval-fusion   # 评测官方 bevfusion-det.pth
```

### 6.2 评测自己训的权重

`eval-fusion` 默认指向官方 `pretrained/bevfusion-det.pth`。评测你自己的 checkpoint：

```bash
torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  runs/run-XXXX/epoch_6.pth \
  --eval bbox \
  --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=2
```

该命令输出 **NDS / mAP 等指标**，默认**不会**出可视化图。

### 6.3 可视化

```bash
torchpack dist-run -np 1 python tools/visualize.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  --mode pred \
  --checkpoint runs/run-XXXX/epoch_6.pth \
  --split val \
  --bbox-score 0.3 \
  --out-dir viz/epoch6 \
  --max-samples 20          # 可选：只画前 N 帧
```

输出目录示例：

```text
viz/epoch6/
  camera-0/ ... camera-5/   # 6 路相机 + 投影 3D 框
  lidar/                    # 俯视点云 BEV + 3D 框
```

- `--mode pred`：画模型预测（置信度 ≥ `--bbox-score`）
- `--mode gt`：画真值框（可不加 `--checkpoint`）
- 类别：nuScenes 10 类（car / truck / bus / … / traffic_cone / barrier）

---

## 7. 目录结构

```text
.
├── install_env.sh          # 一键环境 + 下载 pretrained
├── run_bevfusion.sh        # 一键 demo / 训练 / 评测
├── run_demo_infer.sh       # LiDAR-only 小规模推理
├── env.txt                 # 环境包版本记录
├── configs/                # 训练配置（检测 / 分割）
├── mmdet3d/                # 模型与算子（含本仓库兼容性修补）
├── tools/                  # train / test / visualize / create_data / 下载权重
├── docs/                   # 中文说明文档
├── pretrained/             # 官方权重（需下载，默认不进 Git）
├── data/nuscenes -> ...    # 数据集软链接（本地）
├── runs/                   # 训练日志与 checkpoint（本地）
└── viz/                    # 可视化输出（本地）
```

算法模块关系见：[`docs/02_BEVFusion_算法结构说明.md`](docs/02_BEVFusion_算法结构说明.md)。

---

## 8. 常见问题与已知限制

1. **`torchpack.utils.tqdm` 找不到**  
   已改为 `from tqdm import tqdm`（`tools/visualize.py`）。

2. **`get_cam_feats() takes 3 ... but 4 were given`**  
   DepthLSS 已兼容父类传入的 `mats_dict`。

3. **训练 / 可视化中途 `indice_cuda.cu ... cuda error 9`**  
   个别样本稀疏下采样后无有效体素；已在 spconv 侧做空输出保护，可视化脚本也会跳过失败帧继续。

4. **`Expected more than 1 value per channel when training`（BN）**  
   batch=1 时稀疏特征过少；建议 `BATCH=2`，并已加稀疏 BN / 体素填充兜底。

5. **完整 val 评测中途 CUDA / spconv 失败**  
   与「PyTorch 按 CUDA 11.3 编译、主机驱动 / toolkit 过新」有关。可优先：`samples_per_gpu=1`、只跑 mini、或使用官方建议的 CUDA 11.3 Docker。

6. **Git 不包含大文件**  
   `pretrained/`、`runs/`、`viz/`、`data/`、`*.pth`、`*.so` 等已忽略；协作者需自行下载权重与数据。

---

## 9. 文档

- [`docs/01_nuScenes_数据集说明.md`](docs/01_nuScenes_数据集说明.md)
- [`docs/02_BEVFusion_算法结构说明.md`](docs/02_BEVFusion_算法结构说明.md)

---

## 10. 上游项目与引用

本仓库算法与大量实现来自 MIT Han Lab **BEVFusion**。官方介绍、榜单、Docker、完整训练命令请阅读：

- 代码与 README：https://github.com/mit-han-lab/bevfusion  
- 论文：https://arxiv.org/abs/2205.13542  
- 主页：http://bevfusion.mit.edu/

若本工作或上游 BEVFusion 对你有帮助，请引用原论文：

```bibtex
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

上游还基于 / 受益于 mmdetection3d、LSS、BEVDet、TransFusion、CenterPoint 等开源工作，详见[官方 Acknowledgements](https://github.com/mit-han-lab/bevfusion/blob/master/README.md)。
