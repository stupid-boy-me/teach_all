# MobileNetV2

> V1 会省算力；V2 更会「倒着走残差」：中间胖、两头瘦，并对最后一层激活格外克制（Linear Bottleneck）。

论文：[MobileNetV2](https://arxiv.org/abs/1801.04381)

## 阅读重点

1. Inverted Residual：为何 1×1 先升维再深度卷积？  
2. 为何某些分支去掉 ReLU？  
3. Expand ratio 对速度/精度的影响

训练脚本与数据路径以本目录文件为准；指标可视化可参考同级 [`../MobileNetv3_with_Confusion_Matrix`](../MobileNetv3_with_Confusion_Matrix)。
