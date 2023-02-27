import torch
import torch.nn as nn
from torchsummary import summary
'''
link:https://blog.csdn.net/weixin_48167570/article/details/120688156
'''
class Conv(nn.Module):
    def __init__(self, c_in, c_out, k, s, p, bias=True):
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
            nn.Conv2d(c_in, c_out, k, s, p),
            #归一化
            nn.BatchNorm2d(c_out),
            #激活函数
            nn.LeakyReLU(0.1),
        )

    def forward(self, entry):
        return self.conv(entry)

class ConvResidual(nn.Module):
    def __init__(self, c_in):		# converlution * 2 + residual
        """
        自定义残差单元，只需给出通道数，该单元完成两次卷积，并进行加残差后返回相同维度的特征图
        :param c_in: 通道数
        """
        c = c_in // 2
        super(ConvResidual, self).__init__()
        self.conv = nn.Sequential(
            Conv(c_in, c, 1, 1, 0),		 # kernel_size = 1进行降通道
            Conv(c, c_in, 3, 1, 1),		 # 再用kernel_size = 3把通道升回去
        )
        
    def forward(self, entry):
        return entry + self.conv(entry)	 # 加残差，既保留原始信息，又融入了提取到的特征
        # 采用 1*1 + 3*3 的形式加深网络深度，加强特征抽象

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)			# 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64)				# 一个残差块 = 2层卷积
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(				# = 4层卷积
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26 = nn.Sequential(				# = 16层卷积
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
        )
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43 = nn.Sequential(				# = 16层卷积
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
        )
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52 = nn.Sequential(				# = 8层卷积
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
        )

    def forward(self, entry):
	    conv1 = self.conv1(entry)
	    conv2 = self.conv2(conv1)
	    conv3_4 = self.conv3_4(conv2)
	    conv5 = self.conv5(conv3_4)
	    conv6_9 = self.conv6_9(conv5)
	    conv10 = self.conv10(conv6_9)
	    conv11_26 = self.conv11_26(conv10)
	    conv27 = self.conv27(conv11_26)
	    conv28_43 = self.conv28_43(conv27)
	    conv44 = self.conv44(conv28_43)
	    conv45_52 = self.conv45_52(conv44)
	    return conv45_52, conv28_43, conv11_26		# YOLOv3用，所以输出了3次特征
		

def darknet53():
    return Darknet53().to("cuda")


if __name__ == "__main__":
    x = torch.randn(1,3,224,224).cuda()
    model_darknet53 = darknet53()
    print(len(model_darknet53(x)))
    # summary(model_densenet,(3,224,224))