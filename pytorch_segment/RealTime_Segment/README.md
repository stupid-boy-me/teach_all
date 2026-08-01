# RealTime_Segment —— 实时语义分割

> 目标：在「还算实时」的预算里，给每个像素一个类别。  
> 本目录提供训练、数据加载与 ONNX 预测的基础骨架，文档会持续加厚。

---

## 目录结构

```text
RealTime_Segment/
├── train.py            # 训练入口
├── dataset.py          # Dataset
├── predict_onnx.py     # ONNX 推理示例
├── models/             # 网络定义
├── utils/              # 工具函数
└── scripts/            # 辅助脚本
```

---

## 快速开始（草案）

```bash
# 1) 准备好分割数据集（如 Cityscapes），并按 dataset.py 期望的目录摆好
# 2) 按需修改 train.py 中的路径与超参
python train.py

# ONNX 推理（导出模型后）
python predict_onnx.py
```

具体超参、数据路径、预训练权重说明：**待补充**（你本地跑通一版后欢迎回填到本 README）。

---

## 建议数据

见上级 [`../README.md`](../README.md) 与仓库根网盘表（Cityscapes / CDLA）。

---

## 下一步文档计划

- [ ] 数据目录约定与标签颜色表  
- [ ] 训练命令与常用超参  
- [ ] mIoU 评估脚本说明  
- [ ] 导出 ONNX 的注意事项  

先把链路跑通，再追求漂亮曲线——分割项目尤其如此。
