# vision —— 视觉组件参考区

> 有时候你需要的不是完整训练工程，而是「这个 Backbone 长什么样」。  
> 本目录偏向 **torchvision 风格的参考实现 / 积木**，方便抄结构、对接口。

---

## 结构

| 路径 | 内容 |
|------|------|
| [`torchvison/`](./torchvison) | 数据与模型参考（目录名保留历史拼写） |
| [`torchvison/data`](./torchvison/data) | 数据相关 |
| [`torchvison/models`](./torchvison/models) | 模型相关 |
| [`torchvison/models/classification`](./torchvison/models/classification) | 分类骨干集合（AlexNet、DarkNet、DenseNet、EfficientNet、HRNet、CSPNet…） |

更完整的「可训练工程」请优先看 [`../pytorch_classifier`](../pytorch_classifier)。

---

## 使用建议

1. 当速查手册：打开对应 `*.py` 看网络定义  
2. 需要完整 train/val 循环时，去 `pytorch_classifier` 找同系列工程  
3. 发现接口不统一或缺文档——欢迎 PR，文档债我们一起还

---

## 状态

建设中。README 会随代码丰富逐步补「模型对照表 + 参数量笔记」。
