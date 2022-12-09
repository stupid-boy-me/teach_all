import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchsummary import summary
 
'''
 参考链接: https://blog.csdn.net/qq_57886603/article/details/121531939
'''
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
 
#基本的卷积BNrelu
class baseConv(nn.Module):
    def __init__(self,inchannels,outchannels,kernel_size,stride,groups,hasRelu=False):
        super(baseConv, self).__init__()
        if hasRelu:
            #判断是否有relu激活函数
            activate=nn.ReLU
        else:
            activate=nn.Identity
        pad=kernel_size//2
        self.baseconv=nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=outchannels,kernel_size=kernel_size,stride=stride,padding=pad,groups=groups,bias=False),
            nn.BatchNorm2d(outchannels),
            activate()
        )
 
    def forward(self,x):
        out=self.baseconv(x)
        return out
 
 
#通道重排
def ChannelShuffle(x,groups):
    batch_size,channel,height,width=x.size()
    #获得每组的组内channel
    inner_channel=channel//groups
    #[batch,groups,inner_channel,height,width]
    x=x.view(batch_size,groups,inner_channel,height,width)
    x=torch.transpose(x,1,2).contiguous()
    x=x.view(batch_size,-1,height,width)
    return x
 
 
#stage结构
class Residual(nn.Module):
    def __init__(self,inchannels,outchannels,stride,groups):
        super(Residual, self).__init__()
        self.add_=True      #shortcut为相加操作
        self.groups=groups
 
        hidden_channel=inchannels// 4
        #当输入channel不等于24时候才有第一个1*1conv
        self.has_conv1=True
        if inchannels!=24:
            self.channel1_first1=baseConv(inchannels=inchannels,outchannels=hidden_channel,kernel_size=1,stride=1,groups=groups,hasRelu=True)
        else:
            self.has_conv1=False
            self.channel1_first1=nn.Identity()
            hidden_channel=inchannels
 
        #channel1
        self.channel1=nn.Sequential(
            baseConv(inchannels=hidden_channel,outchannels=hidden_channel,kernel_size=3,stride=stride,groups=hidden_channel),
            baseConv(inchannels=hidden_channel,outchannels=outchannels,kernel_size=1,stride=1,groups=groups)
        )
 
        #channel2
        if stride==2:
            self.channel2=nn.AvgPool2d(kernel_size=3,stride=stride,padding=1)
            self.add_=False
 
 
    def forward(self,x):
        if self.has_conv1:
            x1=self.channel1_first1(x)
            x1=ChannelShuffle(x1,groups=self.groups)
            out=self.channel1(x1)
        else:
            out=self.channel1(x)
        if self.add_:
            out+=x
            return F.relu_(out)
        else:
            out2=self.channel2(x)
            out=torch.cat((out,out2),dim=1)
            return F.relu_(out)
 
 
#shuffleNet
class ShuffleNet(nn.Module):
    def __init__(self,groups,out_channel_list,num_classes,rate,init_weight=True):
        super(ShuffleNet, self).__init__()
 
        #定义有序字典存放网络结构
        self.Module_List=OrderedDict()
 
        self.Module_List.update({'Conv1':nn.Sequential(nn.Conv2d(3,_make_divisible(24*rate,divisor=4*groups),3,2,1,bias=False),nn.BatchNorm2d(_make_divisible(24*rate,4*groups)),nn.ReLU())})
        self.Module_List.update({'MaxPool1':nn.MaxPool2d(3,2,1)})
 
        #net_config [inchannels,outchannels,stride]
        net_config=[[out_channel_list[0],out_channel_list[0],1],
                    [out_channel_list[0],out_channel_list[1],2],
                    [out_channel_list[1],out_channel_list[2],1],
                    [out_channel_list[2],out_channel_list[3],2],
                    [out_channel_list[3],out_channel_list[4],1]]
        repeat_num=[3,1,7,1,3]
 
        #搭建stage部分
        self.Module_List.update({'stage0_0':Residual(_make_divisible(24*rate,4*groups),_make_divisible((out_channel_list[0]-_make_divisible(24*rate,4*groups))*rate,4*groups),stride=2,groups=groups)})
        for idx,item in enumerate(repeat_num):
            config_item=net_config[idx]
            for j in range(item):
                if j==0 and idx!=0 and config_item[-1]==2:
                    self.Module_List.update({'stage{}_{}'.format(idx,j+1):Residual(_make_divisible(config_item[0]*rate,4*groups),_make_divisible((config_item[1]-config_item[0])*rate,4*groups),config_item[2],groups)})
                else:
                    self.Module_List.update({'stage{}_{}'.format(idx,j+1):Residual(_make_divisible(config_item[0]*rate,4*groups),_make_divisible(config_item[1]*rate,4*groups),config_item[2],groups)})
                config_item[-1]=1       #重复stage的stride=1
                config_item[0]=config_item[1]
 
        self.Module_List.update({'GlobalPool':nn.AvgPool2d(kernel_size=7,stride=1)})
 
        self.Module_List=nn.Sequential(self.Module_List)
 
        self.linear=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(_make_divisible(out_channel_list[-1]*rate,4*groups),num_classes)
        )
 
        if init_weight:
            self.init_weight()
    def forward(self,x):
        out=self.Module_List(x)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out
 
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
 
#定义shufflenet_
def shuffleNet_g1_(num_classes,rate=1.0):
    config=[144,288,288,576,576]
    return ShuffleNet(groups=1,out_channel_list=config,num_classes=num_classes,rate=rate)
 
def shuffleNet_g2_(num_classes,rate=1.0):       #
    config=[200,400,400,800,800]
    return ShuffleNet(groups=2,out_channel_list=config,num_classes=num_classes,rate=rate)
 
def shuffleNet_g3_(num_classes,rate=1.0):
    config=[240,480,480,960,960]
    return ShuffleNet(groups=3,out_channel_list=config,num_classes=num_classes,rate=rate)
 
def shuffleNet_g4_(num_classes,rate=1.0):
    config=[272,544,544,1088,1088]
    return ShuffleNet(groups=4,out_channel_list=config,num_classes=num_classes,rate=rate)
 
def shuffleNet_g8_(num_classes,rate=1.0):
    config=[384,768,768,1536,1536]
    return ShuffleNet(groups=8,out_channel_list=config,num_classes=num_classes,rate=rate)
 
if __name__ == '__main__':
    model_shuffleNet_g3_=shuffleNet_g3_(10,rate=1.0).to('cuda')
    # print(model_shuffleNet_g3_)
    summary(model_shuffleNet_g3_,(3,224,224))
