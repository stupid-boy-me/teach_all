# pth / pt → ONNX

> PyTorch 权重走出实验室的第一站，常常是 ONNX。  
> 这一步做得干净，后面的 `onnx2ncnn` 才会少骂人。

本目录放置导出脚本与说明。原有详细笔记保留如下（若有路径请改成你的本机路径）。

---

## 导出时建议记住的三件事

1. **`model.eval()`**，并关闭不需要的 dropout / 随机分支  
2. 输入用 `torch.randn` 时，**尺寸要与线上一致**（含 batch=1）  
3. 明确 `opset_version`、`input_names` / `output_names`，方便下游工具识别

```python
import torch

model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
)
```

导出后可用 Netron 打开 `.onnx` 目视检查；再用 onnxruntime 跑一遍与 PyTorch 对齐。

接下来：回到 [`../README.md`](../README.md) 走 NCNN 转换与量化。

---

## 历史笔记

## 模型转换
步骤1：[pth模型转换成onnx模型代码](https://github.com/stupid-boy-me/teach_all/blob/main/NCNN/NCNN%E9%87%8F%E5%8C%96%E5%B7%A5%E5%85%B7/pth(pt)2onnx.py/pth2onnx.py)

步骤2：onnx进行简化

第一步：安装onnx_simplifier

```Python
pip install onnx-simplifier

```

第二步：onnx文件简化

```Python
python -m onnxsim --skip-optimization A.onnx B_sim.onnx
```

步骤3：onnx模型转成ncnn模型

```Python
onnx2ncnn.exe A.onnx A.param A.bin
```

步骤4：ncnn模型优化

```Python
ncnnoptimize.exe A.param A.bin A_opt.param A_opt.bin 0

```

步骤5：ncnn2mem

```Python
ncnn2mem.exe A_sim.param A_sim.bin A_sim.id.h A_sim.mem.h
```

ncnn模型转换地址：

D:\nextvpu\yanye\model_convert\ncnn

参考链接：[https://zhuanlan.zhihu.com/p/391519043](https://zhuanlan.zhihu.com/p/391519043)


