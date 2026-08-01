# 1.format_convert —— 检测标注格式转换

> 模型可以一周换一个，标注格式能陪你熬三个通宵。  
> 本目录放 VOC / YOLO 等常见格式的互转小工具，专治「数据明明有，脚本却读不进去」。

---

## 使用前三问

1. 你的标注现在是：**XML（VOC）**、**txt（YOLO）**，还是别的？  
2. 训练脚本期望哪一种？  
3. 类别名 ↔ id 的映射表是否一致？（不一致时 mAP 会用一种很安静的方式惩罚你）

---

## 建议用法

```bash
# 先备份原始标注！
# 再按脚本参数修改路径后执行（以目录内实际文件为准）
python your_convert_script.py
```

转换后抽查几张：用可视化工具或检测工程自带的 draw 脚本看框是否还在物体上。

---

## 相关

- YOLOv1 数据侧：[`../yolov1/data2voc`](../yolov1/data2voc)  
- Faster R-CNN / VOC：[`../faster_rcnn`](../faster_rcnn)

后续会补：支持格式清单、参数说明、踩坑 FAQ。
