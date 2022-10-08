import torch
import torch.nn as nn


class Bottleneck(nn.Module): # Convblock
    def __init__(self, in_channel, filters, s):
        super(Bottleneck, self).__init__()
        c1, c2, c3 = filters
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.short_cut = nn.Conv2d(in_channel, c3, kernel_size=1, stride=s, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output_x = self.bottleneck(x)
        short_cut_x = self.batch1(self.short_cut(x))
        result = output_x + short_cut_x
        X = self.relu(result)
        return X


class BasicBlock(nn.Module):
    def __init__(self,in_channel,filters):
        super(BasicBlock, self).__init__()
        c1, c2, c3 = filters
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        output_x = self.basicblock(x)
        X = identity + output_x
        X = self.relu(X)
        return X


class ResNet(nn.Module):
    def __init__(self,num_class):
        super(ResNet, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, filters=[64, 64, 256],s=1),
            BasicBlock(256, filters=[64, 64, 256]),
            BasicBlock(256, filters=[64, 64, 256])
        )

        self.stage3 = nn.Sequential(
            Bottleneck(256, [128, 128, 512],s=2),
            BasicBlock(512, filters=[128, 128, 512]),
            BasicBlock(512, filters=[128, 128, 512]),
            BasicBlock(512, filters=[128, 128, 512]),
        )

        self.stage4 = nn.Sequential(
            Bottleneck(512, [256, 256, 1024],s=2),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
            BasicBlock(1024, filters=[256, 256, 1024]),
        )

        self.stage5 = nn.Sequential(
            Bottleneck(1024, [512, 512, 2048],s=2),
            BasicBlock(2048, filters=[512, 512, 2048]),
            BasicBlock(2048, filters=[512, 512, 2048]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.stage1(x)  # torch.Size([1, 64, 56, 56])
        out = self.stage2(out)  # torch.Size([1, 64, 56, 56])
        out = self.stage3(out) # torch.Size([1, 512, 28, 28])
        out = self.stage4(out) # torch.Size([1, 1024, 14, 14])
        out = self.stage5(out)  # torch.Size([1, 2048, 7, 7])
        out = self.pool(out)
        out = out.view(out.size(0), 2048)
        out = self.fc(out)
        return out


