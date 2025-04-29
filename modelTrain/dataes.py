import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class Mydata(Dataset):
    def __init__(self, lines):
        super(Mydata, self).__init__()
        self.lines = lines
        random.shuffle(self.lines)  # 打乱列表次序，乱序

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        src_path = self.lines[index].strip().split('#')[0]
        lab_path = self.lines[index].strip().split('#')[1]
        img = cv2.imread(src_path)
        lab = cv2.imread(lab_path)
        si = 512
        img = cv2.resize(img, (si, si))
        lab = cv2.resize(lab, (si, si))

        # 归一化
        img = (img / 255.0).astype('float32')
        img = np.transpose(img, (2, 0, 1))
        lab = (lab / 255.0).astype('float32')
        lab = np.transpose(lab, (2, 0, 1))

        img = torch.from_numpy(img)
        lab = torch.from_numpy(lab)
        return img, lab


class Cdata(Dataset):
    def __init__(self, lines):
        super(Cdata, self).__init__()
        self.lines = lines
        random.shuffle(self.lines)  # 打乱列表次序，乱序

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        src_path = self.lines[index].strip().split('#')[0]
        lab = self.lines[index].strip().split('#')[1]
        img = cv2.imread(src_path)
        img = cv2.resize(img, (256, 256))
        img = (img / 255.0).astype('float32')
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        if lab == '0':
            lab = torch.tensor([1.0, 0.0, 0.0])
        elif lab == '1':
            lab = torch.tensor([0.0, 1.0, 0.0])
        else:
            lab = torch.tensor([0.0, 0.0, 1.0])
        return img, lab


if __name__ == '__main__':
    lines = open('train_data.txt', 'r').readlines()
    my = Cdata(lines=lines)
    myloader = DataLoader(dataset=my, batch_size=3, shuffle=False)

    for i, j in myloader:
        print(i.shape, j.shape)
