import torch
import torch.nn as nn
from torchsummary import summary
from torch.hub import load_state_dict_from_url

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

# 定义Fire模块的属性
class Fire(nn.Module):
	# __init__定义参数属性初始化，接着使用super(Fire, self).__init__()调用基类初始化函数对基类进行初始化
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        # inplanes:输入通道 squeeze_planes:输出通道 expand1x1_planes：1x1卷积层 expand3x3_planes：3x3卷积层
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        # inplace=True表示可以在原始的输入上直接操作，不会再为输出分配额外内存，但同时会破坏原始的输入                           
        self.expand1x1_activation = nn.ReLU(inplace=True)
        # 为了使1x1和3x3的filter输出的结果又相同的尺寸，在expand modules中，给3x3的filter的原始输入添加一个像素的边界（zero-padding）.
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        # squeeze 和 expand layers中都是用ReLU作为激活函数
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
    	# 将输入x经过squeeze layer进行卷机操作，再经过squeez_activation()进行激活
        x = self.squeeze_activation(self.squeeze(x))
        # 将输出分别送到expand1x1和expand3x3中进行卷积核激活
        # 最后使用torch.cat()将两个相同维度的张量连接在一起，这里dim=1，代表按列连接，最终得到若干行
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

# version=1.0和version=1.1两个SqueezeNet版本
# version=1.0只有AlexNet的1/50的参数，而1.1在1.0的基础上进一步压缩，计算了降低为1.0的40%左右，主要体现在(1)第一层卷积核7x7更改为3x3 (2)将max_pool层提前了，减少了计算量
class SqueezeNet(nn.Module):
	# num_classes:分类的类别个数
    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
        	# self.features:定义了主要的网络层，nn.Sequential()是PyTorch中的序列容器
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                # ceil_mode=True会对池化结果进行向上取整而不是向下取整
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
        	# 在fire9 module之后，使用Dropout，比例取50%
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            # 使用卷积层代替全连接层
            # 即先采用AdaptiveAvgPool2D,将size变为1x1,channel数=num_classes,再做resize
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            	# 遍历看是否为最后一层，卷积层的初始化方法不同
                if m is final_conv: 
                	# 均值为0，方差为0.01的正太分布初始化方法
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                	# 使用He Kaiming中的均匀分布初始化方法
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
					# 偏置初始化为0
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # 返回一个折叠成一维的数组
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, num_classes=1000):
    model = SqueezeNet(version, num_classes=num_classes)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, num_classes=1000):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, num_classes=1000)


def squeezenet1_1(pretrained=False, progress=True, num_classes=1000):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, num_classes=num_classes)


if __name__ == '__main__':
    model_squeezenet1_0 = squeezenet1_0(pretrained=False, progress=True,num_classes=1000).to("cuda")

    summary(model_squeezenet1_0,(3,224,224))