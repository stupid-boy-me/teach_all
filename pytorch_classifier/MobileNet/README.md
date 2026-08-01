# MobileNet 家族

> 深度可分离卷积：把「又厚又重」的标准卷积拆成「先深度、后逐点」。  
> 手机端爱用它，不是因为名字可爱，是因为 **FLOPs / 参数量** 真的更友好。

---

## 子目录

| 目录 | 说明 | 链接 |
|------|------|------|
| MobileNetV1 | 深度可分离卷积开山 | [MobileNetV1](./MobileNetV1) |
| MobileNetV2 | 倒残差 + Linear Bottleneck | [MobileNetV2](./MobileNetV2) |
| MobileNetV3 | NAS + SE + 激活函数调教 | [MobileNetV3](./MobileNetV3) |
| MobileNetV3 + 混淆矩阵 | 训练与指标可视化一体 | [MobileNetv3_with_Confusion_Matrix](./MobileNetv3_with_Confusion_Matrix) |

**动物数据集**：[百度网盘](https://pan.baidu.com/s/1b0lbd8vOfZcq0V5NyGbroQ)（提取码 `qdbo`）

---

## 学习顺序

```text
V1（会拆卷积）→ V2（会倒残差）→ V3（会看结构搜索痕迹）→ 带混淆矩阵工程（会交付）
```

轻量不等于玩具：部署侧（[`../../NCNN`](../../NCNN)）经常就是这类骨干先上场。
