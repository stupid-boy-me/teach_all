# PyTorch 图像分类

> 分类是视觉入门的「主线任务」：数据管线、Backbone、训练循环、评价指标——四件套练熟了，后面检测/分割才不会慌。

本目录按**模型家族**拆分子文件夹。已落地的有链接；列表里未打勾的是路线图（欢迎催更，更欢迎 PR）。

---

## 学习建议

1. 先跑通 **ResNet50**：建立「数据划分 → Dataset → train/test」肌肉记忆  
2. 再看 **SE-ResNet50**：理解通道注意力只是「加一点点智慧」  
3. 轻量网络三部曲：**MobileNet → ShuffleNet → EfficientNet**（移动端友好）  
4. 配套读 [`Confusion_Matrix`](./Confusion_Matrix)：会训练还不够，会读数才算毕业

公共小工具也可复用上层的 [`../data_split`](../data_split) 与 [`../docs/utils`](../docs/utils)。

---

## 已实现 / 规划中

### ✅ 已有代码（点进去开练）

| 模型 / 模块 | 说明 | 入口 |
|-------------|------|------|
| ResNet50 | 经典残差网络教学实现 | [ResNet50](./ResNet50) |
| SE-ResNet50 | Squeeze-and-Excitation 加持版 | [SE_ResNet50](./SE_ResNet50) |
| MobileNet V1/V2/V3 | 深度可分离卷积家族 | [MobileNet](./MobileNet) |
| MobileNetV3 + 混淆矩阵 | 训练 + 指标可视化一条龙 | [MobileNetv3_with_Confusion_Matrix](./MobileNet/MobileNetv3_with_Confusion_Matrix) |
| ShuffleNet V1/V2 | 通道混洗的效率派 | [ShuffleNet](./ShuffleNet) |
| EfficientNet | 复合缩放思想 | [EfficientNet](./EfficientNet) |
| Confusion Matrix | 分类指标文档 + 代码 | [Confusion_Matrix](./Confusion_Matrix) |

### 🧭 路线图（占位，后续慢慢加）

SE-ResNeXt · RegNet · Swin / ViT / DeiT · RepVGG · ConvNeXt · HRNet · VAN · ConvMixer · CSPNet · PoolFormer · MViT · EfficientFormer · HorNet · MLP-Mixer · Conformer · T2T-ViT · Twins · Res2Net …

> 列表很长不是为了唬人，是为了告诉你：**分类骨干的世界很大，但入门路径可以很短。**

---

## 通用目录习惯（多数子项目类似）

```text
某模型/
├── model.py / model_*.py   # 网络定义
├── train.py                # 训练
├── test.py / predict.py    # 推理
├── data_split*.py          # 划分（或复用上层工具）
└── README.md               # 本模型说明书
```

---

## 数据集

动物分类等网盘链接见[仓库根 README](../README.md#数据集网盘常用)。  
遇到 RGBA 四通道图导致训练炸裂？用各目录里的「判断是否是 RGB」脚本先洗一遍——血泪经验，诚不我欺。

---

## 环境

```bash
conda activate teach_all   # 或你自己的环境
pip install -r ../requirement.txt
```

具体脚本以各子目录 README 为准。祝 train loss 温柔下降，val acc 稳步上升。
