# PyTorch 语义分割

> 分类给整图一个标签；检测给物体画框；分割则认真到**每一个像素**。  
> 标签图一打开，世界突然变成调色盘——欢迎来到「精细活」工种。

---

## 当前内容

| 项目 | 说明 | 状态 |
|------|------|------|
| [`RealTime_Segment`](./RealTime_Segment) | 实时语义分割：训练 / 预测 / ONNX 相关脚本 | ✅ 已有代码 |

### RealTime_Segment 速览

```text
RealTime_Segment/
├── train.py           # 训练入口
├── dataset.py         # 数据加载
├── predict_onnx.py    # ONNX 推理示例
├── models/            # 网络结构
├── utils/             # 工具
└── scripts/           # 辅助脚本
```

详细说明见：[RealTime_Segment/README.md](./RealTime_Segment/README.md)

---

## 推荐数据

| 数据 | 用途 | 网盘 |
|------|------|------|
| Cityscapes | 街景语义分割经典集 | 见[根 README](../README.md#数据集网盘常用) |
| CDLA | 文档版面类分割等 | 同上 |

---

## 学习路线（规划）

```text
数据与标签格式（mask / palette）
        ↓
轻量实时分割（本目录 RealTime_Segment）
        ↓
经典结构：FCN → U-Net → DeepLab 系列（待补充）
        ↓
导出 ONNX → NCNN / 量化部署（见 ../NCNN、../quantization）
```

---

## 环境

分割项目对 `mmcv` / 自定义算子依赖各异，请以子项目脚本为准。通用依赖可参考：

```bash
pip install -r ../requirement.txt
```

---

## 占位：即将到来

- 更完整的训练配置与指标说明（mIoU 面板）  
- U-Net / DeepLabv3+ 教学实现  
- 可视化：预测 mask 叠加原图的一键脚本  

分割不难在「会写 `CrossEntropy`」，难在**数据与可视化闭环**——这块我们慢慢补齐。
