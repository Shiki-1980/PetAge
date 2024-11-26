import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Sequential as Seq
import torch

class DenseBlock(nn.Module):
    out = 1
    def __init__(self, convNum, inchannel, outchannel):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range (convNum):
            layer.append(
                nn.Sequential(
                nn.Conv2d(inchannel+outchannel * i, outchannel, kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(outchannel),
                SELayer(outchannel),
                nn.ReLU(inplace=True))
            )
        self.net = nn.Sequential(*layer)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        for blk in self.net:
            y = blk(x)
            x = torch.cat((x, y), dim=1)
        x = self.dropout(x)
        return x

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

        

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze step
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Global average pooling
        y = self.fc(y).view(b, c, 1, 1)  # Excitation (Fully Connected layers)
        return x * y.expand_as(x)        # Recalibration (scaling feature maps)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

                        

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x                                            # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # 完成一次卷积
        x = self.bn2(self.conv2(x))                             # 第二次卷积不加relu激活函数

        x += identity                                           # 两路相加
        return F.relu(x, inplace=True)                          # 添加激活函数输

class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):        # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(                    # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.se = SELayer(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)                       # 调用change_channel对输入修改，为后面相加做变换准备

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))                             # 完成残差部分的卷积
        x = self.se(x)

        x += identity
        return F.relu(x, inplace=True)                          # 输出卷积单元

class ConvNet(nn.Module):
    def __init__(self, classes_num):
        super(ConvNet, self).__init__()
        self.prepare = nn.Sequential(           # 预处理==》[batch, 64, 56, 56]
            nn.Conv2d(3, 64, 3, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ca1 = SpatialAttention(kernel_size=3)

        self.layer1 = nn.Sequential(            # layer1有点特别，由于输入输出的channel均是64，故两个CommonBlock
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            nn.Dropout(0.2)
        )


        self.layer2 = nn.Sequential(            # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个CommonBlock
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            nn.Dropout(0.2)
        )


        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            nn.Dropout(0.2)
        )


        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            nn.Dropout(0.2)
        )

        self.ca2 = SpatialAttention(kernel_size=3)

        self.head = Seq(nn.AdaptiveAvgPool2d(1),
                  nn.Flatten(start_dim=1),
                  nn.BatchNorm1d(512),
                  nn.Linear(512, classes_num))

    def forward(self, x):
        x = self.prepare(x)         # 预处理

        x = self.ca1(x) * x

        x = self.layer1(x)          # 四个卷积单元

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.ca2(x) * x

        x = self.head(x)

        return x

class DenseNet(nn.Module):
    def __init__(self, classNum=100):
        super(DenseNet, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ca1 = SpatialAttention(kernel_size=3)
        self.blk = []
        self.num_channels = 64
        self.growth_rate = 24
        self.convs_in_dense=[4, 4, 4, 4]
        for i,num in enumerate(self.convs_in_dense):
            self.blk.append(DenseBlock(num, self.num_channels, self.growth_rate ))
            self.num_channels += num*self.growth_rate
            if i!=3:
                self.blk.append(transition_block(self.num_channels, self.num_channels // 2))
                self.num_channels = self.num_channels // 2

        self.course = nn.Sequential(*self.blk)
        self.ca2 = SpatialAttention(kernel_size=3)

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(start_dim=1),
        #     nn.BatchNorm1d(self.num_channels),
        #     nn.Linear(self.num_channels, 100)
        # )
    
    def forward(self, x):
        x = self.prepare(x)
        x = self.ca1(x)*x
        x = self.course(x)
        x = self.ca2(x)*x
        # x = self.head(x)
        return x

config2 = {
    "lr":5e-2,
    "momentum":0.9,
    "weight_decay":5e-4,
}