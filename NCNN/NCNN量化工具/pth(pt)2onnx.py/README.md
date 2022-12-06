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


