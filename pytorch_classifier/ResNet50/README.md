# ResNet50 —— 残差网络入门首选

> 梯度消失曾让网络「越深越废」；残差连接让深度重新变得体面。  
> 如果你只能精读一个分类工程，从这里开始准没错。

论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## 文件与配套视频

| 文件名 | 功能 | 代码 | 视频 |
|--------|------|------|------|
| `data_split2.py` | 划分 train/val/test | [链接](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/ResNet50/data_split2.py) | [BV1dP411P7yx](https://www.bilibili.com/video/BV1dP411P7yx/) |
| `model.py` | ResNet50 模型 | [链接](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/ResNet50/model.py) | [BV1Ye411V7bz](https://www.bilibili.com/video/BV1Ye411V7bz/) |
| `train.py` | 训练脚本 | [链接](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/ResNet50/train.py) | [BV1jW4y1E7dL](https://www.bilibili.com/video/BV1jW4y1E7dL/) |
| `test.py` | 预测脚本 | [链接](https://github.com/stupid-boy-me/teach_all/blob/main/pytorch_classifier/ResNet50/test.py) | [BV1Te4y1e7gb](https://www.bilibili.com/video/BV1Te4y1e7gb/) |
| `resnet.py` | 更完整的 ResNet 家族定义 | 本目录 | — |

**动物数据集**：[百度网盘](https://pan.baidu.com/s/1b0lbd8vOfZcq0V5NyGbroQ)（提取码 `qdbo`，若失效见根 README）

---

## 踩坑提醒

数据集里可能混有 **4 通道 RGBA** 图像。请先用「判断是否是 RGB 格式」类脚本清洗，再训练——否则报错信息可能指东打西。

均值方差可用 [`../../docs`](../../docs) 或本系列脚本计算后写进 normalize。

---

## 建议流程

```text
清洗 RGB → data_split2 → 改 train.py 路径/类别数 → train → test
```

读懂 `model.py` 里的 shortcut 后，再去看 SE-ResNet / MobileNet，会有种「原来大家在残差主题上remix」的畅快感。
