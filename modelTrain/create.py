import torch.nn as nn
import torch
import torchvision.models as models

resnet18 = models.resnet50()
num_ftrs = resnet18.fc.in_features # 获取低级特征维度
resnet18.fc = nn.Linear(num_ftrs, 3) # 替换新的输出层
resnet18.add_module('add_softmax', nn.Softmax(dim=1))

torch.save(resnet18, 'resnet50.pth')

num = 1880
dir_path = "./data/"


def get_txt():
    f = open('class_data.txt', 'w', encoding='utf-8')
    for i in range(1, num):
        src_path = dir_path + "0/" + str(i) + '.jpg'
        lab = '0'
        f.write(src_path + "#" + lab + '\n')
    for i in range(1, num):
        src_path = dir_path + "1/" + str(i) + '.jpg'
        lab = '1'
        f.write(src_path + "#" + lab + '\n')
    for i in range(1, num):
        src_path = dir_path + "2/" + str(i) + '.jpg'
        lab = '2'
        f.write(src_path + "#" + lab + '\n')


if __name__ == '__main__':
    get_txt()
    print('ok')
