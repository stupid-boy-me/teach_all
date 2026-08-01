# NCNN 量化 / 转换工具包（Windows）

> 把「网上说可以转」变成「你双击/命令行真能转」。  
> 本目录汇总 NCNN 在 Windows 环境下常用的可执行工具，配合视频课使用效果最佳。

---

## 工具清单

| 工具 | 典型用途 |
|------|----------|
| `onnx2ncnn.exe` | ONNX → NCNN（`.param` + `.bin`） |
| `ncnnoptimize.exe` | 图优化，去掉多余算子/常量折叠等 |
| `ncnn2table.exe` | 生成量化校准表（int8 前奏） |
| `ncnn2int8.exe` | 转 int8 模型 |
| `ncnn2mem.exe` | 模型嵌入内存等形式转换 |
| `caffe2ncnn.exe` | 老古董 Caffe 模型转换（如仍需要） |

Python 辅助：[`pth(pt)2onnx.py/`](./pth(pt)2onnx.py) —— PyTorch 权重导出 ONNX 的说明与脚本思路。

---

## 推荐流水线

```text
.pth / .pt  →  (Python 导出)  →  .onnx
                                      ↓ onnx2ncnn
                              .param + .bin
                                      ↓ ncnnoptimize（可选）
                              优化后的 NCNN 模型
                                      ↓ ncnn2table + ncnn2int8（可选）
                              int8 模型上端侧
```

---

## 注意事项

1. **先看视频再硬刚**：环境（Vulkan / OpenCV / protobuf）比 exe 本身更容易翻车。见上级 [NCNN README](../README.md) 课表。  
2. ONNX **opset** 与动态维度可能导致转换失败——从静态 shape、常用 opset（如 11）试起。  
3. 量化需要有代表性的**校准图片**；随便截两张表情包当校准集，掉点会很诚实。

微信 `zuiaixiaopaomo`，报错请带完整命令行输出。
