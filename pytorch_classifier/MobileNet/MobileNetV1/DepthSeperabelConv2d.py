import torch.nn as nn
# define DepthSeperabelConv2d with depthwise+pointwise
class DepthSeperabelConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == '__main__':
    model = DepthSeperabelConv2d(16,32,3)
    print(model)