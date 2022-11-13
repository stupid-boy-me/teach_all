# 模型转换

步骤1：pth模型转换成onnx模型代码

```python
import torch
from modeling.deeplab import DeepLab
# from nets.hrnet import HRnet
import torch.onnx
model = DeepLab(num_classes=3,backbone='mobilenet',output_stride=16,sync_bn=None,freeze_bn=False)
# model = HRnet(num_classes = 3, backbone = 'hrnetv2_w18', pretrained = False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
#加载模型参数
checkpoint = torch.load(r'D:\nextvpu\yanye\model_convert\model_best.pth.tar',device)
print(checkpoint.keys())
model.load_state_dict(torch.load(r'D:\nextvpu\yanye\model_convert\model_best.pth.tar',device)['state_dict'])
# 输入
x = torch.randn(1,3,513,513)
# 导出模型
torch_out = torch.onnx._export(model,x,"model_best1.onnx",
                               verbose=True,
                               opset_version=11,
                               training= False,
                               input_names=['input'],
                               output_names=['output'])
print("***********************************************")
print("模型输出成功")
```

步骤2：onnx进行简化

第一步：安装onnx\_simplifier

```python
pip install onnx-simplifier

```

第二步：onnx文件简化

```python
python -m onnxsim --skip-optimization A.onnx B_sim.onnx
```

步骤3：onnx模型转成ncnn模型

```python
onnx2ncnn.exe A.onnx A.param A.bin
```

步骤4：ncnn模型优化

```python
ncnnoptimize.exe A.param A.bin A_opt.param A_opt.bin 0

```

步骤5：ncnn2mem

```python
ncnn2mem.exe A_sim.param A_sim.bin A_sim.id.h A_sim.mem.h
```
