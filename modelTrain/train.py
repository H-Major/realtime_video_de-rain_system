from derainNet import *
from dataes import *
import torch.optim as optim
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot

import os

# ============= 参数设置=======================================

train_data_name = "sight_and_squirrel"  # 数据集目录名字

Epoch = 10  # 训练轮次

Lr = 0.0000005  # 学习率
Gamma = 0.965  # 学习率衰减
Batch_size = 4  # 批量

choose_lose = 5.2  # 选取lose < choose_lose 的模型

restart = 0  # 是否重新训练模型


# ============================================================

def is_file_empty(file_path):
    try:
        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            # 如果文件内容为空，则返回 True
            return not bool(file.read())
    except FileNotFoundError:
        # 如果文件不存在，则返回 False
        print("文件不存在")
        return False


def file_exists(file_path):
    return os.path.exists(file_path)


def read_first_line(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()  # 去除首尾的空白符
    except FileNotFoundError:
        return "文件未找到"
    except Exception as e:
        return f"发生错误: {str(e)}"


def write_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return "写入成功"
    except Exception as e:
        return f"写入失败: {str(e)}"


def create_file(relative_path, file_name):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 组合相对路径和目标文件名
        target_file_path = os.path.join(current_dir, relative_path, file_name)

        # 检查目标文件是否存在
        if not os.path.exists(target_file_path):
            # 如果不存在则创建空文件
            with open(target_file_path, 'w') as file:
                print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下创建成功！")
        else:
            print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下已经存在。")
    except OSError as error:
        print(f"创建文件 '{file_name}' 时出错: {error}")


def create_directory2(relative_path, directory_name):
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 组合相对路径和目标文件夹名称
        target_dir = os.path.join(current_dir, relative_path, directory_name)

        # 检查目标目录是否存在
        if not os.path.exists(target_dir):
            # 如果不存在则创建目录
            os.makedirs(target_dir)
            print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下创建成功！")
        else:
            print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下已经存在。")
    except OSError as error:
        print(f"创建目录 '{directory_name}' 时出错: {error}")


def append_float_to_file(float_num, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(str(float_num) + '\n')
        print("PSNR:", float_num)
    except Exception as e:
        print("写入文件时出错:", str(e))


def read_floats_from_file(file_path):
    float_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 尝试将每一行的内容转换为浮点数并添加到列表中
                try:
                    float_num = float(line.strip())
                    float_list.append(float_num)
                except ValueError:
                    # 如果无法转换为浮点数，则跳过该行
                    pass
        print("成功从文件中读取浮点数。")
        return float_list
    except Exception as e:
        print("读取文件时出错:", str(e))
        return None


def psnr_fun(out, label):
    Psnr = 0.0
    for i in range(len(out)):
        # 将张量转换为 NumPy 数组
        out_np = out[i].cpu().detach().numpy()
        label_np = label[i].cpu().detach().numpy()
        Psnr += psnr(out_np, label_np)
    return Psnr / float(len(out))


def save_Training_history(Y_label, name):
    PSNR = read_floats_from_file(train_data_path + "config/" + "PSNR.txt")
    plt.plot(PSNR, linestyle='-', color='b')  # 训练数据执行结果，『-』表示实线，『b』表示蓝色
    title = 'Training ' + Y_label + ' history'
    plt.title(title)  # 显示图的标题 Training PSNR_SSIM history
    plt.xlabel('Epochs')  # 显示 x 轴标签 epoch
    plt.ylabel(Y_label)  # 显示 y 轴标签 PSNR或者SSIM
    plt.savefig(train_data_path + "training_images/" + name + '.png')
    # plt.show(block=False)  # 开始绘图


if __name__ == '__main__':
    train_data_path = "./" + train_data_name + "/"
    train_data_file = "train_data.txt"  # 数据集文档名
    train_data_doc_path = train_data_path + train_data_file
    choosed_model_path = train_data_path + "choosed_trained_model/"  # loss < choose_lose 的模型
    backup_model_path = train_data_path + "backup_trained_model/"  # 模型备份

    create_directory2(train_data_path, "backup_trained_model")
    create_directory2(train_data_path, "choosed_trained_model")
    create_directory2(train_data_path, "training_images")
    create_directory2(train_data_path, "config")
    create_file(train_data_path, "log.txt")
    create_file(train_data_path + "/config", "PSNR.txt")
    create_file(train_data_path + "/config", "Epoch.txt")
    create_file(train_data_path + "/config", "model_name.txt")

    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DerainModel()

    if not restart:
        retrain_model_name = read_first_line(train_data_path + "config/" + "model_name.txt")
        stadic = torch.load(choosed_model_path + retrain_model_name)
        offset = int(read_first_line(train_data_path + "config/" + "Epoch.txt"))
        model.load_state_dict(stadic)
        print("正在继续训练模型:", retrain_model_name)
        print("Epochs:", offset, "/", offset + Epoch)
    else:
        offset = 0
    model = model.to(device)

    train_lines = open(train_data_doc_path, 'r').readlines()

    num_train = len(train_lines)
    epoch_step = num_train // Batch_size

    # 设置损失函数
    loss_fun = nn.L1Loss()
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=Lr, betas=(0.5, 0.999))
    # 学习率衰减
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Gamma)

    """迭代读取训练数据"""
    train_data = Mydata(train_lines)
    train_loader = DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)

    epoch_step = num_train / Batch_size

    for epoch in range(1, Epoch + 1):
        model.train()
        total_loss = 0
        total_psnr = 0
        with tqdm(total=epoch_step, desc=f'Epoch {epoch}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for step, (features, labels) in enumerate(train_loader, 1):
                features = features.to(device)
                labels = labels.to(device)
                batch_size = labels.size()[0]
                out = model(features)

                loss = loss_fun(out, labels)
                total_psnr += psnr_fun(out, labels)
                # total_ssim += ssim_fun(out, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
                pbar.set_postfix(**{'loss': total_loss.item() / step * 255})
                pbar.update(1)

        # 保存模型
        Loss = total_loss / epoch_step * 255
        model_save_name = (train_data_name +
                           "__Ep-" + str(epoch + offset) +
                           "__Loss-" + str(round(Loss.item(), 2)) +
                           "__PSNR-" + str(round(total_psnr / epoch_step, 2)))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), backup_model_path + model_save_name + ".pth")
        log = (train_data_name +
               "   Epochs " + str(epoch + offset) +
               "   Loss " + str(round(Loss.item(), 4)) +
               "   PSNR " + str(round(total_psnr / epoch_step, 2)) +
               "\n"
               )
        with open(train_data_path + "log.txt", 'a') as f:
            f.write(log)
        print(log)
        append_float_to_file(round(total_psnr / epoch_step, 2), train_data_path + "config/" + "PSNR.txt")
        save_Training_history("PSNR", model_save_name)
        if Loss < choose_lose or epoch == Epoch:
            torch.save(model.state_dict(), choosed_model_path + model_save_name + ".pth")
        write_to_file(train_data_path + "config/" + "model_name.txt", model_save_name + ".pth")
        write_to_file(train_data_path + "config/" + "Epoch.txt", str(epoch + offset))
        lr_scheduler.step()
