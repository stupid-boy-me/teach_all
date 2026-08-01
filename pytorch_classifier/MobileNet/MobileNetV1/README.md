# MobileNetV1

> 核心公式感：`Standard Conv ≈ Depthwise Conv + Pointwise (1×1) Conv`。  
> 先把这套拆解在 `model` 里看明白，再谈 width multiplier / resolution multiplier。

论文：[MobileNets](https://arxiv.org/abs/1704.04861)

## 本目录

以仓库内 `model` / `train` / `test` 等脚本为准。  
数据准备可复用 [`../../../data_split`](../../../data_split) 或 ResNet50 同款划分流程。

## 建议自测

- [ ] 参数量是否明显小于同分辨率 ResNet  
- [ ] 替换 backbone 后训练能否收敛  
- [ ] 导出 ONNX 是否顺利（为部署做准备）

详细超参表与结果记录：**待你跑通后回填**（欢迎 PR）。
