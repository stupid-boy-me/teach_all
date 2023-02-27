import torch.nn as nn
import torch.nn.functional as F
from ShuffleNetv1_Unit import Bottleneck
from torchsummary import summary
class ShuffleNetv1(nn.Module):
    def __init__(self, groups, channel_num, class_num=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self.make_layers(24, channel_num[0], 4, 2, groups)
        self.stage3 = self.make_layers(channel_num[0], channel_num[1], 8, 2, groups)
        self.stage4 = self.make_layers(channel_num[1], channel_num[2], 4, 2, groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_num[2], class_num)
    def make_layers(self, input_channels, output_channels, layers_num, stride, groups):
        layers = []
        layers.append(Bottleneck(input_channels, output_channels - input_channels, stride, groups))
        input_channels = output_channels
        for i in range(layers_num - 1):
            Bottleneck(input_channels, output_channels, 1, groups)
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


#定义shufflenetv1
def shuffleNet_g1_(class_num):
    config=[144,288,576]
    return ShuffleNetv1(groups=1,channel_num=config,class_num=class_num).to('cuda')
def shuffleNet_g2_(class_num):
    config=[200,400,800]
    return ShuffleNetv1(groups=1,channel_num=config,class_num=class_num).to('cuda')
def shuffleNet_g3_(class_num):
    config=[240,480,960]
    return ShuffleNetv1(groups=1,channel_num=config,class_num=class_num).to('cuda')
def shuffleNet_g4_(class_num):
    config=[272,544,1088]
    return ShuffleNetv1(groups=1,channel_num=config,class_num=class_num).to('cuda')
def shuffleNet_g8_(class_num):
    config=[384,768,1536]
    return ShuffleNetv1(groups=1,channel_num=config,class_num=class_num).to('cuda')

if __name__ == '__main__':
    model_shuffleNet_g1 = shuffleNet_g2_(class_num=1000)
    summary(model_shuffleNet_g1,(3,224,224))
    '''
    ps:ShuffleNetv1里面的参数groups=3,channel_num=[240,480,960],class_num=1000)
    是以下这几种情况
    congig = [(1, [144,288,576] , class_num=1000),
              (2, [200,400,800] , class_num=1000),
              (3, [240,480,960] , class_num=1000),
              (4, [272,544,1088], class_num=1000),
              (8, [384,768,1536], class_num=1000)]
    '''