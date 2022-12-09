import torch.nn as nn
import torch
import torch.nn.functional as F

def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    return out
 
 
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
 
        mid_channles = int(out_channels/4)
 
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups
 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channles, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True)
        )
 
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channles, mid_channles, 3, stride=stride, padding=1, groups=mid_channles, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channles, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
 
        self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))
 
        self.stride = stride
 
    def forward(self, x):
 
        out = self.conv1(x)
        out = shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)
 
        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([out, res], 1))
        else:
            out = F.relu(out+x)
 
        return out
if __name__ == '__main__':
    shufflenet = Bottleneck(64,128,3,8)
    print(shufflenet)
