# SE-ResNet50 —— 给残差网装上「通道注意力」

> ResNet 负责「更深也能训」；SE 模块负责「哪些通道更值得听」。  
> 改动不大，思想很香：用全局信息给通道重新加权。

论文：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

---

## 本目录脚本

| 文件 | 作用 |
|------|------|
| `data_splits.py` | 数据集划分 |
| `dataset.py` | 自定义 Dataset（类似 ImageFolder 思路） |
| `model_SE_ResNet50.py` | SE-ResNet50 模型 |
| `train.py` | 训练 |
| `判断是否是RGB格式.py` | 清洗非 RGB 图像 |
| `acquire_mean_std.py` | 统计均值方差 |

---

## 资源链接

| 资源 | 地址 |
|------|------|
| 模型权重 | [百度网盘](https://pan.baidu.com/s/1enZI4CqA87toqUQjiq7E8g)（提取码 `jp1n`） |
| 数据集 | [百度网盘](https://pan.baidu.com/s/1RtJAjr7RXgRdeRZyMqtOAA)（提取码 `j472`） |

---

## 阅读提示

对照 ResNet50，重点看 SE 模块插在何处：`Squeeze`（全局池化）→ `Excitation`（两层 FC + sigmoid）→ 通道缩放。  
参数对齐后先小数据过拟合自检，再上完整集——注意力模块也救不了标签写反。
