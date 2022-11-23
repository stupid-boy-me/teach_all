import torch
import torch.onnx
from model_resnet50 import resnet50
from model_pilipala import resnet18
from model_v2 import MobileNetV2
from model import ResNet
from SE_ResNet50 import SEResNet
# model = resnet50(num_classes=2)
# model = ResNet(num_class=2)
model = MobileNetV2(num_classes=2)
# model = SEResNet(num_class=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
#加载模型参数
checkpoint = torch.load(r'D:\nextvpu\yanye\code\1_analyse_wenlidu\weights_class_C02_C03_600_300.pth',device)
print(checkpoint.keys())
model.load_state_dict(torch.load(r'D:\nextvpu\yanye\code\1_analyse_wenlidu\weights_class_C02_C03_600_300.pth',device)) #['state_dict'])
# 输入
x = torch.randn(1,3,600,300)
# 导出模型
torch_out = torch.onnx._export(model,x,r"D:\nextvpu\yanye\code\1_analyse_wenlidu\weights_class_C02_C03_600_300.onnx",
                               verbose=True,
                               opset_version=11,
                               training= False,
                               input_names=['input'],
                               output_names=['output'])
print("***********************************************")
print("模型输出成功")
