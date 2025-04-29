import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np


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


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        # 卷积层,卷积核数32,尺寸3*3,矩阵四周填充1层0,效果是保持矩阵卷积后长度、宽度不变
        self.conv1 = Double_Conv2d(3, 32)
        # 最大池化层，尺寸2*2，步长2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Double_Conv2d(32, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Double_Conv2d(64, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Double_Conv2d(128, 256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 将多维向量降为二维，如[b,c,h,w]->[b,c*h*w]
        self.out = nn.Sequential(
            nn.Linear(256 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # Sigmoid激活函数，将输出映射到[0,1]的范围内。
        )

    def forward(self, x):  # x.shape=tenosr[1,c,h,w],c=3,(RGB三通道),h=128,w=128
        x = self.conv1(x)  # 经过该层，x.shape=tenosr[batch_size,32,512,512]
        x = self.maxpool1(x)  # 经过该层，x.shape=tenosr[batch_size,32,256,256]
        x = self.conv2(x)  # 经过该层，x.shape=tenosr[batch_size,64,256,256]
        x = self.maxpool2(x)  # 经过该层，x.shape=tenosr[batch_size,64,128,128]
        x = self.conv3(x)  # 经过该层，x.shape=tenosr[batch_size,128,128,128]
        x = self.maxpool3(x)  # 经过该层，x.shape=tenosr[batch_size,128,64,64]
        x = self.conv4(x)  # 经过该层，x.shape=tenosr[batch_size,256,64,64]
        x = self.maxpool4(x)  # 经过该层，x.shape=tenosr[batch_size,256,32,32]
        x = self.flatten(x)
        out = self.out(x)
        return out


# 加载模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('trained_model/clast.pth')
model = model.to(device)
model.eval()


def rainClassify(img):
    img2 = cv2.resize(img, (256, 256))
    img2 = (img2 / 255.0).astype('float32')
    inputs = np.transpose(img2, (2, 0, 1))
    inputs = np.expand_dims(inputs, axis=0)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.to(device)
    out = model(inputs)
    out = out.cpu().detach().numpy()
    return np.argmax(out)
