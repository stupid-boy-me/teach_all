import torch
from torch import nn
from torch.nn import functional as F
from .model_registry import register_model

@register_model()
class FDMobileNet(torch.nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(FDMobileNet, self).__init__()

        def conv_3x3(inp, oup, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                torch.nn.BatchNorm2d(inp),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.ReLU(inplace=True),
            )

        def conv_1x1(inp, oup):
            return torch.nn.Sequential(
                torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup),
                torch.nn.ReLU(inplace=True),
            )

        self.features = torch.nn.Sequential(
            conv_3x3(3, 32, 2),
            conv_dw(32, 32, 2),
            conv_1x1(32, 64),
            conv_dw(64, 64, 2),
            conv_1x1(64, 128),
            conv_dw(128, 128, 1),
            conv_1x1(128, 128),
            conv_dw(128, 128, 2),
            conv_1x1(128, 256),
            conv_dw(256, 256, 1),
            conv_1x1(256, 256),
            conv_dw(256, 256, 2),
            conv_1x1(256, 512),
            conv_dw(512, 512, 1),
            conv_1x1(512, 512),
            conv_dw(512, 512, 1),
            conv_1x1(512, 512),
            conv_dw(512, 512, 1),
            conv_1x1(512, 512),
            conv_dw(512, 512, 1),
            conv_1x1(512, 512),
            conv_dw(512, 512, 1),
            conv_1x1(512, 1024),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # self.pool = torch.nn.AvgPool2d(7)
        # self.fc = torch.nn.Linear(1024, 1000)
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

        # 初始化模型权重
        # if pretrained:
        #     print("Load FDMobileNet Pretrained Weights")
        #     self.load_state_dict(
        #         {
        #             k: v
        #             for k, v in torch.load(
        #                 r"/algdata01/Intern001HF/Work/Lane/Ultra-Fast-Lane-Detection-v2/pre_trained/FDMobileNet_b384x4_e160_a0.6330.pth"
        #             )["model_state_dict"].items()
        #             if k in self.state_dict()
        #         }
        #     )
        # else:
        #     self.apply(self._init_weights)

        # 初始化模型权重
        if pretrained:
            print("Load FDMobileNet Pretrained Weights")
            pretrained_dict = torch.load(
                r"/algdata01/Intern001HF/Work/Lane/Ultra-Fast-Lane-Detection-v2/pre_trained/FDMobileNet_b384x4_e160_a0.6330.pth"
            )["model_state_dict"]
            model_dict = self.state_dict()
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'upsample' not in k and 'output' not in k}
            # Update the new model's state dict
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.output(x)
        # print("x.shape:",x.shape)
        return x


if __name__ == '__main__':

    FDMobileNetmodel = FDMobileNet(2, pretrained=True)
    x = torch.randn(1, 3, 384, 672)
    y = FDMobileNetmodel(x)


