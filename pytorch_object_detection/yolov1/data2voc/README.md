# data2voc —— 把数据弄成 VOC 味道

> 检测项目里，标注格式经常比模型本身更折磨人。  
> 本目录用于把原始标注整理成（或靠近）PASCAL VOC 风格，方便后续转 YOLO 或喂给检测器。

---

## 何时需要它

- 手里是自定义标注（csv / json / 杂乱文件夹）  
- 下游脚本默认吃 VOC（`Annotations/` + `JPEGImages/` + `ImageSets/`）  
- 想先统一格式，再交给 [`../1voc2yolo.py`](../1voc2yolo.py) 或 [`../../1.format_convert`](../../1.format_convert)

---

## 使用说明

以目录内脚本为准；运行前请先改数据路径。  
建议流程：

```text
原始数据 → data2voc → VOC 结构 → （可选）转 YOLO txt
```

文档细化中：示例目录树、字段映射表待补。
