# data_split —— 数据集划分小工具

> 模型再强，也敌不过「训练/验证混在一起」的自我感动。  
> 先把数据分干净，再谈精度——这是性价比最高的一行代码。

本目录放**通用划分与 Dataset 加载**脚本，分类项目里经常直接复用或抄过去改一改。

---

## 脚本说明

| 文件 | 作用 |
|------|------|
| `data_split.py` | 按比例划分成 `train` / `val` / `test` 目录结构 |
| `data_split2.py` | 划分出 path/label 列表（适合自定义 Dataset） |
| `MyDataset.py` | 自定义 Dataset：把样本加载进内存/按需读盘 |
| `mian.py` | 示例主流程：划分 → Dataset → `DataLoader`（文件名是历史遗留彩蛋 😄） |

### 划分前后长什么样？

一开始（按类别堆在一起）：

![data_split](https://user-images.githubusercontent.com/56495543/194759282-b02c3802-92bb-43e7-8dce-764718c722b4.png)

划分之后（train/val/test 井井有条）：

![data_split2](https://user-images.githubusercontent.com/56495543/194759355-bc5b861d-ae1c-4f35-adda-264489f8da50.png)

---

## 推荐用法

```text
原始按类存放的图片
        ↓  data_split.py / data_split2.py
   train / val (/ test) 或 path+label 列表
        ↓  MyDataset.py
        DataLoader → 你的模型
```

比例请按任务改：小数据集别把 val 切到「只剩三张图」——统计上那叫玄学验证。

---

## 相关

- 分类实战里也有类似脚本：[`../pytorch_classifier/ResNet50`](../pytorch_classifier/ResNet50)  
- RGB 清洗、均值方差：[`../docs`](../docs)

后续可补充：分层抽样、k-fold、检测用的 xml/json 同步划分等——有需要提 Issue。
