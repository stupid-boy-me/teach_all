# NCNN 部署专题

> 训练只是恋爱，部署才是婚姻。  
> NCNN 是腾讯开源的高性能神经网络推理框架，尤其擅长**手机 / 嵌入式**这条路。

本目录整理：Windows 下常用工具、官方风格分类 demo，以及配套 B 站视频顺序。  
目标不是「又装一遍环境」，而是把 **pth/pt → ONNX → NCNN → 可执行推理** 这条链路走通。

---

## 目录导航

| 子目录 | 内容 |
|--------|------|
| [`NCNN量化工具`](./NCNN量化工具) | Windows 下常用 exe：`onnx2ncnn`、`ncnnoptimize`、`ncnn2table`、`ncnn2int8`、`ncnn2mem`、`caffe2ncnn` 等 |
| [`官方图像分类代码demo`](./官方图像分类代码demo) | 分类推理完整流程讲解（含模型转换笔记） |

工具说明详见：[NCNN量化工具/README.md](./NCNN量化工具/README.md)

---

## 视频课（建议顺序）

请按下面顺序观看，跳集会变成「玄学装环境」：

| 顺序 | 标题 | 链接 |
|------|------|------|
| ① | lesson9：NCNN 在 Windows 下配置详解 | [BV1TP411w7dG](https://www.bilibili.com/video/BV1TP411w7dG/) |
| ② | lesson14：视频安装讲解（CMake / OpenCV / protobuf / Vulkan / NCNN） | [BV1v84y1v7VE](https://www.bilibili.com/video/BV1v84y1v7VE/) |
| ③ | lesson15：分类推理 demo 代码讲解 | [BV1yg411s7WP](https://www.bilibili.com/video/BV1yg411s7WP/) |
| ④ | lesson16：pth/pt → ONNX → NCNN 逻辑与实战 | [BV1E841187Jp](https://www.bilibili.com/video/BV1E841187Jp/) |

**口诀**：`9 → 14 → 15 → 16`

---

## 典型工作流（心智模型）

```text
PyTorch (.pth / .pt)
        ↓  export
     ONNX (.onnx)
        ↓  onnx2ncnn / 优化
  NCNN (.param + .bin)
        ↓  可选 int8 量化
   端侧可执行 / App
```

细节坑（opset、动态 shape、不支持的算子）请以 demo README 与视频为准——部署的本质是**和算子列表交朋友**。

---

## 相关目录

- 量化资料：[`../quantization`](../quantization)  
- 分类训练代码：[`../pytorch_classifier`](../pytorch_classifier)  

微信：`zuiaixiaopaomo`。若某一步 exe 报错，把完整命令行贴出来，比「转不了」三个字好用一百倍。
