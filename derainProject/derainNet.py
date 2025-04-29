import torch
import torch.nn as nn


# 两层卷积层,分辨率保持不变
class Double_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(Double_Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


# 反卷积层, 将分辨率长宽各扩大一倍
class TransConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=2, padding=1):
        super(TransConv2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, padding=padding, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


# 强化特征图
class SEnet(nn.Module):
    def __init__(self, chs, reduction=4):  # chs（输入特征图的通道数） reduction（降维的比例，默认为4）。
        super(SEnet, self).__init__()
        self.average_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 创建一个自适应平均池化层，将特征图降维到1x1。
        self.fc = nn.Sequential(
            nn.Linear(chs, chs // reduction),  # 第一层线性层，将输入的通道数降低到chs // reduction。
            nn.ReLU(),
            nn.Linear(chs // reduction, chs)  # 第二层线性层，将通道数提升回原来的数值。
        )
        self.activation = nn.Sigmoid()  # 创建一个Sigmoid激活函数层。

    def forward(self, x):  # 前向传播函数。
        ins = x  # 备份输入x。
        batch_size, chs, h, w = x.shape  # 获取输入x的形状，包括批次大小、通道数、高度和宽度。
        x = self.average_pooling(x)  # 对输入x进行自适应平均池化。
        x = x.view(batch_size, chs)  # 将池化后的x改变形状，以便输入到全连接层。
        x = self.fc(x)  # 将x输入到定义的顺序模块中。
        x = self.activation(x)
        x = x.view(batch_size, chs, 1, 1)  # 将处理后的x改变形状，恢复原来的维度。
        return x * ins  # 将处理后的x与原始输入x相乘，然后返回。


n_ch = 40


class DerainModel(nn.Module):
    def __init__(self):
        super(DerainModel, self).__init__()
        # 卷积层,卷积核数32,尺寸3*3,矩阵四周填充1层0,效果是保持矩阵卷积后长度、宽度不变
        self.conv1 = Double_Conv2d(3, n_ch)
        # 最大池化层，尺寸2*2，步长2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Double_Conv2d(n_ch, n_ch * 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Double_Conv2d(n_ch * 2, n_ch * 4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Double_Conv2d(n_ch * 4, n_ch * 8)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.senet = SEnet(chs=n_ch * 8)

        self.conv5 = Double_Conv2d(n_ch * 8, n_ch * 16)

        # 反卷积，将低维数据映射到高维空间。
        self.transConv1 = TransConv2d(n_ch * 16, n_ch * 8)

        self.conv6 = Double_Conv2d(n_ch * 16, n_ch * 8)

        self.transConv2 = TransConv2d(n_ch * 8, n_ch * 4)

        self.conv7 = Double_Conv2d(n_ch * 8, n_ch * 4)

        self.transConv3 = TransConv2d(n_ch * 4, n_ch * 2)

        self.conv8 = Double_Conv2d(n_ch * 4, n_ch * 2)

        self.transConv4 = TransConv2d(n_ch * 2, n_ch)

        self.conv9 = Double_Conv2d(n_ch * 2, n_ch)

        # 实例化一个输出层，该输出层使用一个卷积层和一个Sigmoid激活函数。
        self.out = nn.Sequential(
            nn.Conv2d(n_ch, 32, 1, 1),  # 卷积层，将64维的输入映射到32维的输出。
            nn.Conv2d(32, 3, 1, 1),  # 卷积层，将64维的输入映射到3维的输出。
            nn.Sigmoid()  # Sigmoid激活函数，将输出映射到[0,1]的范围内。
        )

    def forward(self, x):  # x.shape=tenosr[batch_size,3,512,512]
        x1 = self.conv1(x)  # 经过该层，x.shape=tenosr[batch_size,32,512,512]

        x2 = self.maxpool1(x1)  # 经过该层，x.shape=tenosr[batch_size,32,256,256]
        x2 = self.conv2(x2)  # 经过该层，x.shape=tenosr[batch_size,64,256,256]

        x3 = self.maxpool2(x2)  # 经过该层，x.shape=tenosr[batch_size,64,128,128]
        x3 = self.conv3(x3)  # 经过该层，x.shape=tenosr[batch_size,128,128,128]

        x4 = self.maxpool3(x3)  # 经过该层，x.shape=tenosr[batch_size,128,64,64]
        x4 = self.conv4(x4)  # 经过该层，x.shape=tenosr[batch_size,256,64,64]

        x5 = self.maxpool4(x4)  # 经过该层，x.shape=tenosr[batch_size,256,32,32]
        x5 = self.senet(x5)
        x5 = self.conv5(x5)  # 经过该层，x.shape=tenosr[batch_size,512,32,32]

        x6 = self.transConv1(x5)  # 经过该层，x.shape=tenosr[batch_size,512,64,64]
        x6 = torch.cat((x6, x4), dim=1)  # 经过该层，x.shape=tenosr[batch_size,1024,64,64]
        x6 = self.conv6(x6)  # 经过该层，x.shape=tenosr[batch_size,512,64,64]

        x7 = self.transConv2(x6)  # 经过该层，x.shape=tenosr[batch_size,256,128,128]
        x7 = torch.cat((x7, x3), dim=1)  # 经过该层，x.shape=tenosr[batch_size,512,128,128]
        x7 = self.conv7(x7)  # 经过该层，x.shape=tenosr[batch_size,256,128,128]

        x8 = self.transConv3(x7)  # 经过该层，x.shape=tenosr[batch_size,128,256,256]
        x8 = torch.cat((x8, x2), dim=1)  # 经过该层，x.shape=tenosr[batch_size,256,256,256]
        x8 = self.conv8(x8)  # 经过该层，x.shape=tenosr[batch_size,128,256,256]

        x9 = self.transConv4(x8)  # 经过该层，x.shape=tenosr[batch_size,64,512,512]
        x9 = torch.cat((x9, x1), dim=1)  # 经过该层，x.shape=tenosr[batch_size,128,512,512]
        x9 = self.conv9(x9)  # 经过该层，x.shape=tenosr[batch_size,64,512,512]

        out = self.out(x9)  # 经过该层，out.shape=tenosr[batch_size,3,512,512]
        return out




