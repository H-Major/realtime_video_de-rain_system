from derainNet import *
from dataes import *
import torch.optim as optim
from tqdm import tqdm

#加载模型

device = torch.device('cuda:0')
model = torch.load('resnet50.pth')
model = model.to(device)

train_lines = open('class_data.txt', 'r').readlines()

#学习率
lr = 0.00001
#设置batchsize
batch_size = 50

num_train = len(train_lines)
epoch_step = num_train // batch_size

#设置损失函数
loss_fun = nn.CrossEntropyLoss()
#设置优化器
optimizer = optim.Adam(model.parameters(), lr=lr)
#学习率衰减
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

"""迭代读取训练数据"""
train_data = Cdata(train_lines)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    Epoch = 10
    epoch_step = num_train // batch_size
    for epoch in range(1, Epoch + 1):
        model.train()
        total_loss = 0
        total_acc = 0.0
        with tqdm(total=epoch_step, desc=f'Epoch {epoch}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for step, (features, labels) in enumerate(train_loader, 1):
                features = features.to(device)
                labels = labels.to(device)
                batch_size = labels.size()[0]
                out = model(features)
                loss = loss_fun(out, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
                out = out.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                for i, j in zip(out, labels):
                    if np.argmax(i) == np.argmax(j):
                        total_acc += 1.0
                pbar.set_postfix(**{'loss': total_loss.item() / (step),
                                    'acc': total_acc / (step * len(labels))})
                pbar.update(1)
        #保存模型
        Loss = total_loss / (epoch_step)
        Acc = total_acc / ((epoch_step) * batch_size)
        print(Loss, Acc)
        torch.save(model, 'class.pth')
        lr_scheduler.step()
