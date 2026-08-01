# classification backbones（参考实现）

> 这里像零件盒：AlexNet、DarkNet、DenseNet、EfficientNet、HRNet、CSPNet…  
> 适合对照结构；若要「训练到出报表」，优先用 `pytorch_classifier` 里的完整工程。

打开对应 `*.py` 即可阅读网络定义。欢迎补充：

- [ ] 各模型参数量 / 输入尺寸速查表  
- [ ] ImageNet 预训练加载注意事项  
- [ ] 与 `pytorch_classifier` 目录的映射关系

读码愉快；改结构前先跑一遍 `forward` shape 检查。
