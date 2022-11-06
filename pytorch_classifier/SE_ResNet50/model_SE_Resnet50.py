import torch.nn as nn
import torch
class SELayer(nn.Module):
    def __init__(self,in_channel,reduction): # reduction 减少
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel,in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )


    def forward(self,x): # torch.Size([1, 256, 64, 64])
        b,c,_,_ = x.size()
        # x1 = self.avg_pool(x) # torch.Size([1, 256, 1, 1])
        x2 = self.avg_pool(x).view(b,c) # torch.Size([1, 256])
        # print(x2.shape)
        # y1 = self.fc(x2) # torch.Size([1, 256])
        y2 = self.fc(x2).view(b,c,1,1)  # torch.Size([1, 256, 1, 1])
        # print(y2)
        y3 = y2.expand_as(x)
        # print(y3)
        y4 = x * y3
        # print(y4.shape) # torch.Size([1, 256, 64, 64])
        return y4



class Bottleneck(nn.Module):
    def __init__(self,in_channel,filters,stride=2):
        super(Bottleneck, self).__init__()
        c1,c2,c3 = filters
        self.out_channels = c3
        self.in_channel = in_channel
        self.stride = stride
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,out_channels=c1,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            nn.Conv2d(c1,c2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c2,c3,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=c3,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(c3)
        )

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(c3, 16)
    def forward(self,x):
        identity = x
        if self.stride != 1 or self.in_channel != self.out_channels:
            identity = self.downsample(x)
        x = self.bottleneck(x)
        x = self.se(x)
        x += identity
        x = self.relu(x)
        # print(x.shape)
        return x

class SEResNet(nn.Module):
    def __init__(self,num_class):
        super(SEResNet, self).__init__()
        self.channels = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,self.channels,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(self.channels,[64,64,256],stride=1),
            Bottleneck(256,[64,64,256],stride=1),
            Bottleneck(256, [64, 64, 256], stride=1),
            # SELayer(256,16)
        )

        self.stage3 = nn.Sequential(
            Bottleneck(256,[128,128,512],stride=2),
            Bottleneck(512, [128, 128, 512], stride=1),
            Bottleneck(512, [128, 128, 512], stride=1),
            Bottleneck(512, [128, 128, 512], stride=1),
            # SELayer(512,32)
        )

        self.stage4 = nn.Sequential(
            Bottleneck(512, [256, 256, 1024], stride=2),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            Bottleneck(1024, [256, 256, 1024], stride=1),
            # SELayer(1024,64)
        )

        self.stage5 = nn.Sequential(
            Bottleneck(1024,[512,512,2048],stride=2),
            Bottleneck(2048,[512,512,2048],stride=1),
            Bottleneck(2048, [512, 512, 2048], stride=1),
            # SELayer(2048,128)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.fc = nn.Sequential(
            nn.Linear(2048,num_class)
        )
    def forward(self,x):
        x = self.stage1(x)
        # print(x.shape)
        x = self.stage2(x)
        # print(x.shape)
        x = self.stage3(x)
        # print(x.shape)
        x = self.stage4(x)
        # print(x.shape)
        x = self.stage5(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0],2048) # 2048
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1,3,224,224)
    resnet50 = SEResNet(1000)
    y = resnet50(x)
