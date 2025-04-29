from derainNet import *
from dataes import *
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtCore
from PyQt5.QtGui import *
from UI_Train import Ui_MainWindow
from datetime import datetime
import time
import torch.nn.functional as F

'''
python -m PyQt5.uic.pyuic UI_Train.ui -o UI_Train.py

pyinstaller -w -D -i ./img.ico Train_Model.py --hidden-import dataes.py --hidden-import derainNet.py --hidden-import UI_Train.py 
'''


def MSFRLoss(input_tensor, target_tensor):
    label_fft1 = torch.fft.fft2(target_tensor, dim=(-2, -1))
    label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

    pred_fft1 = torch.fft.fft2(input_tensor, dim=(-2, -1))
    pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)
    return F.l1_loss(pred_fft1, label_fft1)


def format_time(seconds):
    if seconds < 3600:
        return time.strftime("%M:%S", time.gmtime(seconds))
    else:
        return time.strftime("%H:%M:%S", time.gmtime(seconds))


def estimate_remaining_time(total_iterations, current_iteration, start_time):
    elapsed_time = time.time() - start_time
    iterations_done = current_iteration + 1
    estimated_total_time = (elapsed_time / iterations_done) * total_iterations
    remaining_time = estimated_total_time - elapsed_time
    formatted_elapsed_time = format_time(elapsed_time)
    formatted_remaining_time = format_time(remaining_time)
    print(f"已经花费时间: {formatted_elapsed_time}, 预估剩余时间: {formatted_remaining_time}")
    return formatted_elapsed_time, formatted_remaining_time


def format_float(number, n):
    return '{:.{}f}'.format(number, n)


def cvimg_to_qtimg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qtimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    return qtimg


def file_exists(file_path):
    return os.path.exists(file_path)


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


def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M")


def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()  # 去除首尾的空白符
    except FileNotFoundError:
        return "文件未找到"
    except Exception as e:
        return f"发生错误: {str(e)}"


def write_config(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
            file.close()
        return "写入成功"
    except Exception as e:
        return f"写入失败: {str(e)}"


def psnr_fun(out, label):
    Psnr = 0.0
    for i in range(len(out)):
        # 将张量转换为 NumPy 数组
        out_np = out[i].cpu().detach().numpy()
        label_np = label[i].cpu().detach().numpy()
        Psnr += psnr(out_np, label_np)
    return Psnr / float(len(out))


def append_log(float_num, file_path):
    try:
        with open(file_path, 'a') as file:
            file.write(str(float_num) + '\n')
        print("add to :" + file_path + ": " + float_num)
    except Exception as e:
        print(file_path + "写入文件时出错:", str(e))


def make_plot(loss_file_path, psnr_file_path, output_image_path):
    with open(loss_file_path, 'r') as f:
        loss_data = [float(line.strip()) for line in f.readlines()]
    with open(psnr_file_path, 'r') as f:
        psnr_data = [float(line.strip()) for line in f.readlines()]
    epochs = list(range(1, len(loss_data) + 1))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axes[0].plot(epochs, loss_data, label='Loss', color='blue')
    axes[0].set_title('Loss——Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(epochs, psnr_data, label='PSNR', color='blue')
    axes[1].set_title('PSNR——Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR')
    axes[1].grid(True)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_image_path)


class TrainModel(QMainWindow, Ui_MainWindow):
    # 构造函数
    def __init__(self):
        super(TrainModel, self).__init__()
        self.setupUi(self)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DerainModel()

        self.train_data_ready = 0

        self.offset = 0
        self.pre_loss = 0
        self.pre_psnr = 0
        self.cur_epoch = 0

        self.Epoch = 0
        self.Lr = 0
        self.Gamma = 0
        self.Batch_size = 0

        self.trained = 0
        self.keep_train_model_path = ""  # 最后一个模型路径

        self.last_image = ""  # 最后一张图像的路径

        self.started = 0
        self.stop = 0
        self.immediate_stop = 0

        self.train_data_path = ""  # 数据集根目录
        self.backup_model_path = ""  # 模型保存目录
        self.train_image_path = ""  # 图片保存目录
        self.log_path = ""  # 日志目录
        self.doc_path = ""  # 数据集txt文档目录

        self.config_path = ""  # 配置、记录保存目录
        self.epoch_path = ""
        self.loss_path = ""
        self.psnr_path = ""
        self.lr_path = ""
        self.gamma_path = ""
        self.trained_model_path = ""  # 存储 最后一个模型的路径 的文件
        self.trained_image = ""  # 存储 最后一张图像的路径 的文件

        self.psnr_log_path = ""  # psnr 记录文件
        self.loss_log_path = ""  # loss 记录文件

        # ======== 信号槽连接 ====================================
        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_stop.clicked.connect(self.immediate_stop_clicked)
        self.check_stop.stateChanged.connect(self.stop_checked)
        self.btn_choose.clicked.connect(self.choose_clicked)

        self.window_init()

    def start_train(self):
        self.Print("开始训练")
        app.processEvents()
        # 加载模型
        if self.trained == 1:
            self.Print("继续训练模型 " + self.keep_train_model_path)
            stadic = torch.load(self.keep_train_model_path)
            self.model.load_state_dict(stadic)
        self.model = self.model.to(self.device)
        print("模型加载成功")
        print("加载数据集：", self.doc_path)
        train_lines = open(self.doc_path, 'r').readlines()
        print(train_lines[0])
        print(train_lines[1])
        print("数据集加载成功")
        num_train = len(train_lines)
        loss_fun = nn.L1Loss()  # 设置损失函数
        print("设置损失函数成功")
        optimizer = optim.Adam(self.model.parameters(),  # =设置优化器
                               lr=self.Lr,
                               betas=(0.5, 0.999))
        print("optimizer设置成功")
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,  # 学习率衰减
                                                 step_size=1,
                                                 gamma=self.Gamma)
        print("lr_scheduler设置成功")
        train_data = Mydata(train_lines)  # 迭代读取训练数据
        print("train_data加载设置成功")
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=self.Batch_size,
                                  shuffle=True)
        print("train_loader设置成功")

        epoch_step = num_train // self.Batch_size
        print("正在进入训练循环")
        for epoch in range(1, self.Epoch + 1):
            self.cur_epoch = epoch + self.offset
            self.Print("正在训练轮次： " + str(self.cur_epoch) + "/" + str(self.Epoch + self.offset))
            print("训练轮次：", epoch + self.offset)
            self.model.train()
            total_loss = 0
            total_psnr = 0
            self.la_cur_epoch.setText("Epoch:" +
                                      str(epoch + self.offset) + "/" +
                                      str(self.Epoch + self.offset))
            app.processEvents()
            start_time = time.time()
            passed_time = ""
            remain_time = ""
            for step, (features, labels) in enumerate(train_loader, 1):
                print("Step:" + str(step) + "/" + str(epoch_step))
                features = features.to(self.device)
                labels = labels.to(self.device)
                out = self.model(features)

                # loss = loss_fun(out, labels)
                l1 = loss_fun(out, labels)
                f1 = MSFRLoss(out, labels)
                loss = l1 + 0.5 * f1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # total_loss += loss
                total_loss += l1
                total_psnr += psnr_fun(out, labels)
                cur_loss = round((total_loss.item() / step * 255), 4)
                print("cur_loss:", cur_loss)
                cur_psnr = round((total_psnr / step), 2)
                print("cur_psnr:", cur_psnr)
                progress = int((step / epoch_step) * 100)
                print("progress:", epoch_step)
                self.la_cur_loss.setText("Loss:" + str(cur_loss))
                self.la_cur_psnr.setText("PSNR:" + str(cur_psnr))
                self.progress.setValue(progress)
                if step < epoch_step - 1:
                    passed_time, remain_time = estimate_remaining_time(epoch_step, step, start_time)
                    self.la_time.setText(passed_time + " < " + remain_time)

                app.processEvents()
                if self.immediate_stop:
                    self.Print("已停止训练")
                    self.started = 0
                    return

            # 更新轮次
            self.la_pre_epoch.setText("当前已训练轮次:" + str(epoch + self.offset))
            write_config(self.epoch_path, str(epoch + self.offset))
            app.processEvents()

            # 保存loss与PSNR日志
            self.pre_loss = round(total_loss.item() / epoch_step * 255, 2)
            self.pre_psnr = round(total_psnr / epoch_step, 2)
            append_log(str(self.pre_loss), self.loss_log_path)
            write_config(self.loss_path, str(self.pre_loss))
            print("Loss", str(self.pre_loss))
            append_log(str(self.pre_psnr), self.psnr_log_path)
            write_config(self.psnr_path, str(self.pre_psnr))
            print("PSNR:", str(self.pre_psnr))
            self.la_pre_psnr.setText("PSNR:" + str(self.pre_psnr))
            self.la_pre_loss.setText("Loss:" + str(self.pre_loss))
            app.processEvents()

            # 更新学习率
            self.Lr = format_float(float(self.Lr) * self.Gamma, 9)
            print("NEW-Lr:", self.Lr)
            write_config(self.lr_path, str(self.Lr))
            print("Gamma:", self.Gamma)
            write_config(self.gamma_path, str(self.Gamma))
            self.line_lr.setText(str(self.Lr))
            app.processEvents()

            # 保存模型
            model_save_name = ("Ep-" + str(epoch + self.offset) +
                               "__Loss-" + str(self.pre_loss) +
                               "__PSNR-" + str(self.pre_psnr))
            self.keep_train_model_path = self.backup_model_path + model_save_name + ".pth"
            torch.save(self.model.state_dict(), self.keep_train_model_path)
            write_config(self.trained_model_path, self.keep_train_model_path)

            # 保存日志
            log = ("Epochs " + str(epoch + self.offset) +
                   "   Loss " + str(self.pre_loss) +
                   "   PSNR " + str(self.pre_psnr)
                   )
            append_log(log, self.log_path)
            self.Print("完成当前轮次训练  " + log + "   用时 " + passed_time)
            app.processEvents()

            # 绘制图像并显示
            self.last_image = self.train_image_path + model_save_name + ".jpg"
            make_plot(self.loss_log_path, self.psnr_log_path, self.last_image)
            write_config(self.trained_image, self.last_image)
            self.show_image()

            lr_scheduler.step()

            if self.stop:
                self.Print("已经停止训练")
                break
        self.Print("训练结束 " +
                   "  Epoch: " + str(self.cur_epoch) +
                   "  Loss: " + str(self.pre_loss) +
                   "  PSNR: " + str(self.pre_psnr)
                   )
        self.started = 0

    def load_parameter(self):
        self.Print("正在加载参数")
        self.Lr = float(self.line_lr.text())
        self.Gamma = float(self.line_gamma.text())
        self.Batch_size = int(self.line_batch.text())
        self.Epoch = int(self.line_epoch.text())
        self.offset = int(read_config(self.epoch_path))
        self.Print("Epoch:" + str(self.Epoch) +
                   "  LearningRate:" + str(self.Lr) +
                   "  Gamma:" + str(self.Gamma) +
                   "  BatchSize:" + str(self.Batch_size)
                   )

    def start_clicked(self):  # 开始训练
        if self.train_data_ready == 0:
            QMessageBox.warning(None, "警告", "请先加载数据集目录!")
        else:
            if self.started == 1:
                self.Print("正在训练中")
            else:
                self.started = 1
                self.stop = 0
                self.immediate_stop = 0
                self.Print("准备开始训练")
                self.load_parameter()
                self.start_train()
                self.check_stop.setChecked(0)

    def load_path(self):
        self.backup_model_path = self.train_data_path + "backup_trained_model/"
        self.train_image_path = self.train_data_path + "training_images/"
        self.doc_path = self.train_data_path + "train_data.txt"
        self.log_path = self.train_data_path + "log.txt"
        self.config_path = self.train_data_path + "config/"
        self.epoch_path = self.config_path + "Epoch.txt"
        self.loss_path = self.config_path + "loss.txt"
        self.psnr_path = self.config_path + "PSNR.txt"
        self.lr_path = self.config_path + "lr.txt"
        self.gamma_path = self.config_path + "gamma.txt"
        self.trained_model_path = self.config_path + "model_name.txt"
        self.trained_image = self.config_path + "image_name.txt"
        self.loss_log_path = self.config_path + "loss_log.txt"
        self.psnr_log_path = self.config_path + "psnr_log.txt"

    def choose_clicked(self):
        '''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "请选择数据集目录", options=options)
        '''
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取数据集目录")
        if directory:
            self.Print("准备加载数据集目录")
            directory = self.to_relative_path(directory)
            if file_exists(directory + "/train_data.txt"):
                # 是数据集目录
                self.train_data_path = directory + "/"
                self.load_path()
                self.la_data_path.setText(self.train_data_path)
                self.train_data_ready = 1
                if file_exists(self.train_data_path + "log.txt") and not is_file_empty(self.train_data_path + "log.txt"):
                    # 已经初始化过
                    self.Print("准备加载数据集信息")
                    self.load_config()
                else:
                    self.Print("准备初始化数据集")
                    self.init_config()
                    self.load_config()
            else:
                self.Print("使用了错误的目录或数据集文件已损坏，请重新选择")

    def load_config(self):  # 加载数据集配置
        self.Print("正在加载数据集信息")
        self.offset = int(read_config(self.epoch_path))
        print("offset加载成功", self.offset)
        self.Lr = read_config(self.lr_path)
        print("Lr加载成功", self.Lr)
        self.Gamma = float(read_config(self.gamma_path))
        print("Gamma加载成功", self.Gamma)

        if self.offset == 0:
            self.Print("该数据集还未经训练")
            self.trained = 0
        else:
            self.Print("该数据集此前已经被训练过")
            self.trained = 1
            self.pre_loss = float(read_config(self.loss_path))
            self.pre_psnr = float(read_config(self.psnr_path))
            print("已加载loss、psnr")
            self.la_pre_loss.setText("Loss:" + str(self.pre_loss))
            self.la_pre_psnr.setText("PSNR:" + str(self.pre_psnr))
            app.processEvents()
            self.keep_train_model_path = read_config(self.trained_model_path)
            self.last_image = read_config(self.trained_image)
            self.show_image()
            app.processEvents()

        self.la_pre_epoch.setText("当前已训练轮次:" + str(self.offset))
        self.line_lr.setText(str(self.Lr))
        self.line_gamma.setText(str(self.Gamma))

    def init_config(self):
        self.Print("正在初始化数据集")
        self.create_directory(self.backup_model_path)
        self.create_directory(self.config_path)
        self.create_directory(self.train_image_path)
        self.create_file(self.log_path)
        self.create_file(self.epoch_path)
        self.create_file(self.lr_path)
        self.create_file(self.gamma_path)
        self.create_file(self.loss_path)
        self.create_file(self.psnr_path)
        self.create_file(self.trained_model_path)
        self.create_file(self.trained_image)
        self.create_file(self.loss_log_path)
        self.create_file(self.psnr_log_path)

        write_config(self.epoch_path, "0")
        write_config(self.lr_path, "0.0001")
        write_config(self.gamma_path, "0.99")

    def show_image(self):
        print("正在加载图像")
        image = cv2.imread(self.last_image)
        image = cvimg_to_qtimg(image)
        pixmap = QPixmap.fromImage(image)
        self.window.setPixmap(pixmap)

    def window_init(self):
        self.setWindowTitle("去雨模型训练工具")
        self.setWindowIcon(QIcon('./_internal/img.ico'))
        self.setStyleSheet("background-color: white;")
        self.la_data_path.setText("")
        self.window.setScaledContents(True)
        self.window.setText("")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("QProgressBar {"
                                    "border: 2px solid black;"
                                    "border-radius: 5px;"
                                    "background-color: #FAFAFA;"
                                    "}"
                                    "QProgressBar::chunk {"
                                    "background-color: #5F5F5F;"
                                    "}")
        self.line_lr.setStyleSheet("""
            QLineEdit {
                border: 2px solid #999999; /* 边框颜色和粗细 */
                border-radius: 4px; /* 圆角半径 */
                padding: 5px; /* 内边距 */
            }
        """)
        self.line_batch.setStyleSheet("""
            QLineEdit {
                border: 2px solid #999999; /* 边框颜色和粗细 */
                border-radius: 4px; /* 圆角半径 */
                padding: 5px; /* 内边距 */
            }
        """)
        self.line_epoch.setStyleSheet("""
            QLineEdit {
                border: 2px solid #999999; /* 边框颜色和粗细 */
                border-radius: 4px; /* 圆角半径 */
                padding: 5px; /* 内边距 */
            }
        """)
        self.line_gamma.setStyleSheet("""
            QLineEdit {
                border: 2px solid #999999; /* 边框颜色和粗细 */
                border-radius: 4px; /* 圆角半径 */
                padding: 5px; /* 内边距 */
            }
        """)
        self.btn_choose.setStyleSheet("""
            QPushButton {
                border: 2px solid #666666; /* 边框颜色和宽度 */
                border-radius: 5px; /* 圆角半径 */
                padding: 5px 10px; /* 内边距 */
                background-color: #FAFAFA;
            }
            QPushButton:hover {
                border: 3px solid #666666; /* 鼠标悬停时的边框颜色和宽度 */
                background-color: #FFFFFF;
            }
        """)
        self.btn_start.setStyleSheet("""
            QPushButton {
                border: 2px solid #666666; /* 边框颜色和宽度 */
                border-radius: 5px; /* 圆角半径 */
                padding: 5px 10px; /* 内边距 */
                background-color: #FAFAFA;
            }
            QPushButton:hover {
                border: 3px solid #666666; /* 鼠标悬停时的边框颜色和宽度 */
                background-color: #FFFFFF;
            }
        """)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                border: 2px solid #666666; /* 边框颜色和宽度 */
                border-radius: 5px; /* 圆角半径 */
                padding: 5px 10px; /* 内边距 */
                background-color: #FAFAFA;
            }
            QPushButton:hover {
                border: 3px solid #666666; /* 鼠标悬停时的边框颜色和宽度 */
                background-color: #FFFFFF;
            }
        """)
        self.text_info.setStyleSheet("""
            QTextEdit {
                border: 2px solid #888888; /* 边框颜色和宽度 */
                border-radius: 4px; /* 圆角半径 */
                padding: 5px; /* 内边距 */
                background-color: #FAFAFA;
            }
        """)
        font = QFont()
        font.setPointSize(11)
        font.setFamily("黑体")
        self.line_gamma.setFont(font)
        self.line_batch.setFont(font)
        self.line_epoch.setFont(font)
        self.line_lr.setFont(font)

    def stop_checked(self):  # 提前结束
        if self.check_stop.isChecked():
            self.stop = 1
            self.Print("本轮训练完成后结束训练")
        else:
            self.stop = 0
            self.Print("取消提前结束")

    def immediate_stop_clicked(self):  # 立即结束
        flag = QMessageBox.question(None, "警告", "是否立即停止训练？", QMessageBox.Yes | QMessageBox.No)
        if flag == QMessageBox.Yes:
            self.immediate_stop = 1
            self.Print("立即停止训练")

    def Print(self, info):
        time = get_current_time()
        info = "[" + time + "] " + info
        print(info)
        self.text_info.insertPlainText(info + "\n")
        self.text_info.verticalScrollBar().setValue(self.text_info.verticalScrollBar().maximum())

    def create_directory(self, directory_name):
        try:
            # 检查目录是否已存在
            if not os.path.exists(directory_name):
                # 如果不存在则创建目录
                os.makedirs(directory_name)
                self.Print(f"'{directory_name}' 创建成功")
                return 0
            else:
                self.Print(f"目录 '{directory_name}' 已经存在。")
                return 1
        except OSError as error:
            self.Print(f"创建目录 '{directory_name}' 时出错: {error}")

    def create_file(self, file_name):
        try:
            # 检查目标文件是否存在
            if not os.path.exists(file_name):
                # 如果不存在则创建空文件
                with open(file_name, 'w') as file:
                    self.Print(f"'{file_name}' 创建成功")
            else:
                self.Print(f"文件 '{file_name}' 已经存在。")
        except OSError as error:
            self.Print(f"创建文件 '{file_name}' 时出错: {error}")

    def to_relative_path(self, path):
        current_work_dir = str(os.path.dirname(__file__)).replace('\\', '/') + '/'
        if "_internal/" in current_work_dir:
            current_work_dir = current_work_dir.replace("_internal/", "")
        self.Print("当前工作目录：" + current_work_dir)
        if current_work_dir in path:
            path = path.replace(current_work_dir, "")
            print("已将绝对路径转换为相对路径：" + path)
        return path


if __name__ == '__main__':  # 运行主函数
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    Window = TrainModel()
    Window.show()

    sys.exit(app.exec_())
