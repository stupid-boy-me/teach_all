# Grad-CAM —— 看看模型在看哪里

> 准确率很高，但模型盯着水印分类？  
> Grad-CAM 用一张热力图告诉你：它到底在看猫，还是在看角落里的 JPEG 伪影。

本目录用于存放 Grad-CAM 相关脚本与说明（持续补充）。

## 典型用途

- 排查「捷径学习」（shortcut learning）  
- 课件/答辩可视化  
- 对比不同 backbone 的关注差异

## 使用提示

需要能拿到目标层的梯度与激活；不同模型 hook 位置不同。  
跑通后请把「推荐 hook 层」写回本 README，方便后来人。

参考阅读可自行搜索：Grad-CAM (Selvaraju et al.)。
