import torch
import torch.onnx
from model_v2 import MobileNetV2

model = MobileNetV2(num_classes=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
#加载模型参数
checkpoint = torch.load(r'你的pth模型',device)
print(checkpoint.keys())
model.load_state_dict(checkpoint) #['state_dict'])
# 输入
x = torch.randn(1,3,800,400)  # 你的模型训练输入尺寸大小
# 导出模型
torch_out = torch.onnx._export(model,x,r"onnx模型输出地址", # 例如 D:\nextvpu\yanye\code\1_analyse_wenlidu\model_C03_F03.onnx
                               verbose=True,
                               opset_version=11,
                               training= False,
                               input_names=['input'],
                               output_names=['output'])
print("***********************************************")
print("模型输出成功")
