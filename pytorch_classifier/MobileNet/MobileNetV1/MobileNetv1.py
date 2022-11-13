import torch.nn as nn
from DepthSeperabelConv2d import  DepthSeperabelConv2d
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
'''
参考链接：https://blog.csdn.net/u010712012/article/details/94888053
'''
class MobileNetV1(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """
    def __init__(self, width_multiplier=1, class_num=1000):
        super().__init__()
        alpha = width_multiplier
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(alpha*32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(alpha*32)),
            nn.ReLU6(inplace=True)
        )
        self.features = nn.Sequential(
            DepthSeperabelConv2d(int(alpha * 32), int(alpha * 64), 1),
            DepthSeperabelConv2d(int(alpha * 64), int(alpha * 128), 2),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 128), 1),
            DepthSeperabelConv2d(int(alpha * 128), int(alpha * 256), 2),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 256), 1),
            DepthSeperabelConv2d(int(alpha * 256), int(alpha * 512), 2),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 512), 1),
            DepthSeperabelConv2d(int(alpha * 512), int(alpha * 1024), 2),
            DepthSeperabelConv2d(int(alpha * 1024), int(alpha * 1024), 2)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(alpha * 1024), class_num)
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cuda()    #input = torch.rand(1,3,224,224).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)

    t2 = time.time()
    for i in range(10):
        model(input)
    t3 = time.time()
    
    torch.save(model.state_dict(), "test_%s.pth"%name)
    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    resnet18 = models.resnet18(num_classes=2).cuda()
    alexnet = models.alexnet(num_classes=2).cuda()
    vgg16 = models.vgg16(num_classes=2).cuda()
    squeezenet = models.squeezenet1_0(num_classes=2).cuda()
    mobilenet = MobileNetV1().cuda()

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet, 'mobilenet')