# Faster R-CNN（教学版 · 默认联合训练）

> 两阶段检测的古典贵族：先提议「哪里可能有东西」，再精细分类与修框。  
> 本实现默认**联合训练**，让你先跑通再回头对照论文四步法——先会开飞机，再拆发动机。

面向「熟悉代码 + 能跑通训练」的实现。  
论文：[*Faster R-CNN*](https://arxiv.org/abs/1506.01497)

**当前默认训练策略：近似联合训练**（`mode=full`，每个 iter 四项 loss 一起回传，无 Step1~4）。  
四步交替脚本仍保留，供对照论文学习。

默认数据/骨干：**PASCAL VOC 2007** + **ResNet-50**。  
`small_subset: 64` 方便先跑通；正式训练改为 `null` 并加大 `joint_epochs`。

---

## 文档

全部在 [`docs/`](./docs/README.md)：

| 路径 | 内容 |
|------|------|
| [docs/结构三件套/](./docs/结构三件套/README.md) | ① Backbone ② RPN ③ RoI Head |
| [docs/代码模块讲解/](./docs/代码模块讲解/README.md) | 对照 `models/*.py` |
| [docs/损失函数.md](./docs/损失函数.md) | 四项 loss |
| [docs/四步训练/](./docs/四步训练/README.md) | 交替 vs 联合、mode |
| [docs/面试速记.md](./docs/面试速记.md) | 口述 |

---

## 训练策略说明

| 策略 | 入口 | 行为 |
|------|------|------|
| **联合（推荐默认）** | `python train.py` 或 `train_joint.py` | 每个 iter：①→②→③，四项 loss 相加，①②③ 一起更新 |
| 四步交替（论文） | `python train_alternating.py` | 先整段 Step1，再 Step2… 分阶段冻层 |

联合训练每个 iter 的损失：

`loss_objectness + loss_rpn_box_reg + loss_classifier + loss_box_reg`

---

## 快速开始（联合训练）

使用 conda 环境 **`fasterrcnn`**。数据默认根目录：`/workspace/datasets`（对应 `E:\WSL\wsl_datasets`）。  
详细路径说明（含 `D:\data\VOCdevkit` 如何接到容器）：[`docs/训练指南.md`](./docs/训练指南.md)。

```bash
conda activate fasterrcnn
cd /workspace/project/codes/object_Detection/faster_rcnn
pip install -r requirements.txt   # 如需
python smoke_test.py

# 缺数据时先下载（只需一次）
python download_voc.py --root /workspace/datasets

# ★ 训练
./train.sh
# 等价: python train.py --config configs/default.yaml
# 可选: ./train.sh --epochs 6

python infer.py --checkpoint outputs/joint/joint_final.pth --image /path/to.jpg
```

数据目录：`/workspace/datasets/VOCdevkit/VOC2007/`（`data.root: /workspace/datasets`）。  
若你已有 `D:\data\VOCdevkit`，请复制或联接到 `E:\WSL\wsl_datasets\VOCdevkit`（容器访问不到 D 盘）。

---

## 目录与文件职责

```
faster_rcnn/
├── train.py / train_joint.py   # ★ 默认：联合训练
├── train_alternating.py        # 论文四步（学习用）
├── configs/default.yaml
├── models/ datasets/ utils/ docs/
└── infer.py / evaluate.py / smoke_test.py
```

### 根目录（入口与脚本）

| 文件 | 干什么 |
|------|--------|
| `train.py` | 默认训练入口，内部转调联合训练 |
| `train_joint.py` | **联合训练**实现：每个 iter 四项 loss 一起更新 |
| `train_alternating.py` | **四步交替训练**（论文风格，学习用） |
| `train.sh` / `run_train.sh` | 一键启动训练的 shell 包装 |
| `engine.py` | 训练/简单评估的**公共循环**（`train_one_epoch`、`evaluate_simple`） |
| `evaluate.py` | 在验证集上跑粗评（检出比例、平均框数；非完整 mAP） |
| `infer.py` | **单图推理可视化**，画框保存图片 |
| `smoke_test.py` | 冒烟测试：快速检查模型能否前向/反传 |
| `download_voc.py` | 下载 VOC 数据到配置的 root |
| `requirements.txt` | Python 依赖列表 |
| `README.md` | 工程总说明（本文件） |

### `configs/`

| 文件 | 干什么 |
|------|--------|
| `default.yaml` | 数据路径、模型超参、训练/评估默认配置 |

### `models/`（网络三件套）

| 文件 | 干什么 |
|------|--------|
| `backbone.py` | ① 共享 Backbone（ResNet C4 特征，stride=16） |
| `rpn.py` | ② RPN：Anchor 生成 + RPNHead + proposals/NMS + RPN loss |
| `roi_head.py` | ③ RoI Head：RoIAlign + FC + 多类分类/类相关回归 + 后处理 |
| `faster_rcnn.py` | 整网组装：`FasterRCNN`、`mode`/`set_trainable`、`build_model` |
| `__init__.py` | 包初始化 |

### `datasets/`

| 文件 | 干什么 |
|------|--------|
| `voc.py` | VOC 数据集读取、类别、构建 train/val |
| `__init__.py` | 包初始化 |

### `utils/`

| 文件 | 干什么 |
|------|--------|
| `boxes.py` | 框 IoU / encode-decode / clip / NMS 辅助 / Smooth L1 等 |
| `transforms.py` | 预处理：短边缩放、归一化等 |
| `misc.py` | 配置加载、checkpoint、collate、冻层、AverageMeter 等杂项 |
| `__init__.py` | 包初始化 |

### `docs/`（讲解文档，不参与训练）

| 路径 | 干什么 |
|------|--------|
| `docs/README.md` | 文档索引与阅读顺序 |
| `训练指南.md` | 环境、路径、如何跑通训练 |
| `损失函数.md` | 四项 loss 说明 |
| `面试速记.md` | 口述要点 |
| `结构三件套/` | ①②③ 概念与代码精讲；含维度图 `网络结构与维度变换.md` 与 `assets/` 导出图 |
| `代码模块讲解/` | 对照 `models/*.py` 的模块讲解 |
| `四步训练/` | 交替 vs 联合、`mode` 说明 |

---

## 说明

- `batch_size=1`；评估为快速检查，非完整 VOC mAP  
- Backbone 为 ResNet（非论文 VGG），结构对应关系不变  
- 现代检测库默认也是联合训练：[torchvision](https://docs.pytorch.org/vision/stable/models/faster_rcnn.html) / Detectron2
