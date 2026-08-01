# quantization —— 模型量化

> 量化的浪漫在于：用更少的比特，换差不多的精度，以及老板喜欢的延迟数字。  
> 本目录目前以**资源汇集 + 占位**为主，完整代码与教程会陆续搬进来。

---

## 当前资源

| 资源 | 说明 | 链接 |
|------|------|------|
| 花数据集 `flower_photos` | 轻量分类练手 | [百度网盘](https://pan.baidu.com/s/1yzR6qXJXEYZ7v9067MxtWw)（提取码 `xyi8`） |
| 预训练 `vgg16NetBest.pth` | 示例权重 | [百度网盘](https://pan.baidu.com/s/1HhXhSJHZxzfralfcmKyZug)（提取码 `8fjg`） |

---

## 和仓库其他部分的关系

```text
pytorch_classifier / segment 训练出 float 模型
                ↓
        ONNX / 校准集
                ↓
     PTQ / QAT（本目录未来重点）
                ↓
        NCNN int8 等端侧格式  →  ../NCNN
```

Windows 下 NCNN 量化相关 exe 已放在 [`../NCNN/NCNN量化工具`](../NCNN/NCNN量化工具)。

---

## 规划中的章节

- [ ] PyTorch 动态/静态量化最小可运行示例  
- [ ] 校准集怎么选、怎么评估掉点  
- [ ] 与 ONNX Runtime / NCNN 的对接笔记  
- [ ] 「量化前后速度/精度」对比表模板  

量化不是魔法，是**误差预算管理**。等示例代码就位后，这里会变成动手手册，而不仅是网盘目录。
