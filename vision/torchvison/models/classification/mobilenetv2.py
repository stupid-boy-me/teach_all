import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary
 
 
#把channel变为8的整数倍 所有层的通道数都可以被 8 整除
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
 
 
#定义基本的ConvBN+Relu
class baseConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,groups=1,stride=1):
        super(baseConv, self).__init__()
        pad=kernel_size//2
        relu=nn.ReLU6(inplace=True)
        if kernel_size==1 and in_channels>out_channels:
            relu=nn.Identity()
        self.baseConv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=pad,groups=groups,bias=False),
            nn.BatchNorm2d(out_channels),
            relu
        )
 
    def forward(self,x):
        out=self.baseConv(x)
        return out
 
 
#定义残差结构
class residual(nn.Module):
    def __init__(self,in_channels,expand_rate,out_channels,stride):         #输入和输出channel都要调整到8的整数倍
        super(residual, self).__init__()
        expand_channel=int(expand_rate*in_channels)     #升维后的channel
 
        conv1=baseConv(in_channels, expand_channel, 1, stride=stride)
        if expand_rate == 1:
            #此时没有1*1卷积升维
            conv1=nn.Identity()
 
        #channel1
        self.block1=nn.Sequential(
            conv1,
            baseConv(expand_channel,expand_channel,3,groups=expand_channel,stride=stride),
            baseConv(expand_channel,out_channels,1)
        )
 
        if stride==1 and in_channels==out_channels:
            self.has_res=True
        else:
            self.has_res=False
 
    def forward(self,x):
        if self.has_res:
            return self.block1(x)+x
        else:
            return self.block1(x)
 
 
#定义mobilenetv2
class MobileNet_v2(nn.Module):
    def __init__(self,theta=1,num_classes=10,init_weight=True):
        super(MobileNet_v2, self).__init__()
        #[inchannel,t,out_channel,stride]
        net_config=[[32,1,16,1],
                    [16,6,24,2],
                    [24,6,32,2],
                    [32,6,64,2],
                    [64,6,96,1],
                    [96,6,160,2],
                    [160,6,320,1]]
        repeat_num=[1,2,3,4,3,3,1]
 
        module_dic=OrderedDict()
 
        module_dic.update({'first_Conv':baseConv(3,_make_divisible(theta*32),3,stride=2)})
 
        for idx,num in enumerate(repeat_num):
            parse=net_config[idx]
            for i in range(num):
                module_dic.update({'bottleneck{}_{}'.format(idx,i+1):residual(_make_divisible(parse[0]*theta),parse[1],_make_divisible(parse[2]*theta),parse[3])})
                parse[0]=parse[2]
                parse[-1]=1
 
        module_dic.update({'follow_Conv':baseConv(_make_divisible(theta*parse[-2]),_make_divisible(1280*theta),1)})
        module_dic.update({'avg_pool':nn.AdaptiveAvgPool2d(1)})
 
        self.module=nn.Sequential(module_dic)
 
        self.linear=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(_make_divisible(theta*1280),num_classes)
        )
        #初始化权重
        if init_weight:
            self.init_weight()
 
    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)
 
 
    def forward(self,x):
        out=self.module(x)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out
 
def mobilenetv2(theta=1,num_classes=1000,init_weight=True):
    
    return MobileNet_v2(theta=theta,num_classes=num_classes,init_weight=init_weight).to("cuda")
 
if __name__ == '__main__':
    model_mobilenetv2 = MobileNet_v2(theta=1).to("cuda")
    summary(model_mobilenetv2,(3,224,224))
