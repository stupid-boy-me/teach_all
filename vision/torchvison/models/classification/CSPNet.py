# CSPNet不仅仅是一个网络，更是一个处理思想，可以和ResNet、ResNext、DenseNet、EfficientNet等网络结合。
# 从实验结果来看，分类问题中，使用CSPNet可以降低计算量，但是准确率提升很小；
# 在目标检测问题中，使用CSPNet作为Backbone带来的提升比较大，可以有效增强CNN的学习能力，同时也降低了计算量
    
import torch
import torch.nn as nn
# no success
class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s, bias=True):
        '''
        自定义一个卷积块，一次性完成卷积+归一化+激活，这在类似于像DarkNet53这样的深层网络编码上可以节省很多代码
        :param c_in: in_channels，输入通道
        :param c_out: out_channels，输出通道
        :param k: kernel_size，卷积核大小
        :param s:  stride，步长
        :param p: padding，边界扩充
        :param bias: …
        '''
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            #卷积
            nn.Conv2d(c_in, c_out, k, s),
            #归一化
            nn.BatchNorm2d(c_out),
            #激活函数
            nn.LeakyReLU(0.1),
        )

    def forward(self, entry):
        return self.conv(entry)

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        print(c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
 
    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
Csp = BottleneckCSP(3,16)#输出通道必须为偶数，如16

x = torch.rand(1,3,608,608)
print(Csp(x).shape)#[1, 16, 608, 608]