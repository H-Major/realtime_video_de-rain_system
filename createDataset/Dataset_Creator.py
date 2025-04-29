from PyQt5.QtGui import QIcon

from UI_createDataset import *
import cv2
import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
from PyQt5 import QtCore

'''
python -m PyQt5.uic.pyuic UI_createDataset.ui -o UI_createDataset.py

pyinstaller -w -D -i E:\Video_Derain_Project\createDataset\icon.ico  Dataset_Creator.py --hidden-import UI_createDataset.py
'''


def to_relative_path(path):
    current_work_dir = str(os.path.dirname(__file__)).replace('\\', '/') + '/'
    if current_work_dir in path:
        path = path.replace(current_work_dir, "")
        print("已将绝对路径转换为相对路径：" + path)
    return path


class CreateDataset(QMainWindow, Ui_Form):
    # 构造函数
    def __init__(self):
        super(CreateDataset, self).__init__()
        self.setupUi(self)

        self.setWindowTitle("图像去雨数据集制作工具")
        self.setWindowIcon(QIcon("./_internal/icon.ico"))

        self.dataset_name = ""
        self.norain_video = ""
        self.rain_video = ""
        self.norain_pth = ""
        self.rain_pth = ""
        self.frame_step = 0
        self.char = ''
        self.data_file = ""
        self.current_work_dir = os.getcwd()

        self.stop = 1

        self.btn_rain.clicked.connect(self.choose_rain_video)
        self.btn_norain.clicked.connect(self.choose_norain_video)
        self.btn_start.clicked.connect(self.start_click)
        self.btn_stop.clicked.connect(self.stop_click)

        self.setWindowTitle('图像去雨数据集制作工具')  # 窗口名称

    def stop_click(self):
        self.stop = 1

    def create_directory1(self, directory_name):
        try:
            # 检查目录是否已存在
            if not os.path.exists(directory_name):
                # 如果不存在则创建目录
                os.makedirs(directory_name)
                self.Print(f"目录 '{directory_name}' 创建成功！")
                return 0
            else:
                self.Print(f"目录 '{directory_name}' 已经存在。")
                return 1
        except OSError as error:
            self.Print(f"创建目录 '{directory_name}' 时出错: {error}")

    def create_directory2(self, relative_path, directory_name):
        try:
            # 获取当前工作目录
            current_dir = os.getcwd()
            # 组合相对路径和目标文件夹名称
            target_dir = os.path.join(current_dir, relative_path, directory_name)

            # 检查目标目录是否存在
            if not os.path.exists(target_dir):
                # 如果不存在则创建目录
                os.makedirs(target_dir)
                self.Print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下创建成功！")
            else:
                self.Print(f"目录 '{directory_name}' 在相对路径 '{relative_path}' 下已经存在。")
        except OSError as error:
            self.Print(f"创建目录 '{directory_name}' 时出错: {error}")

    def create_file(self, relative_path, file_name):
        try:
            # 获取当前工作目录
            current_dir = os.getcwd()
            # 组合相对路径和目标文件名
            target_file_path = os.path.join(current_dir, relative_path, file_name)

            # 检查目标文件是否存在
            if not os.path.exists(target_file_path):
                # 如果不存在则创建空文件
                with open(target_file_path, 'w') as file:
                    self.Print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下创建成功！")
            else:
                self.Print(f"文件 '{file_name}' 在相对路径 '{relative_path}' 下已经存在。")
        except OSError as error:
            self.Print(f"创建文件 '{file_name}' 时出错: {error}")

    def choose_norain_video(self):
        pth, file = QFileDialog.getOpenFileName()
        if pth:
            pth = to_relative_path(pth)
            self.pth_norain.setText(pth)
            self.norain_video = pth

    def choose_rain_video(self):
        pth, file = QFileDialog.getOpenFileName()
        if pth:
            pth = to_relative_path(pth)
            self.pth_rain.setText(pth)
            self.rain_video = pth

    def save_frames(self, norain_video, rain_video, save_pth_1, save_pth_2, data, X, char):
        cap1 = cv2.VideoCapture(norain_video)
        cap2 = cv2.VideoCapture(rain_video)
        i = 1
        counter = 1
        self.Print("正在制作数据集 (可以随时停止制作)")
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            if X == 0 or i % X == 0:
                cv2.imwrite(save_pth_1 + str(counter) + '.jpg', frame1)
                cv2.imwrite(save_pth_2 + str(counter) + '.jpg', frame2)
                with open(data, 'a') as f:
                    f.write(save_pth_1 + str(counter) + '.jpg' + char + save_pth_2 + str(counter) + '.jpg' + '\n')
                counter += 1
                self.num_frame.setText("帧数统计: " + str(counter))
                app.processEvents()
            i += 1
            if self.stop == 1:
                self.Print("\n提前结束数据集制作，共" + str(counter) + "帧图片，已保存到" + self.dataset_name + "目录下")
                return
        self.Print("数据集制作完成，共" + str(counter) + "帧图片，已保存到" + self.dataset_name + "目录下")
        cap1.release()
        cap2.release()

    def Print(self, info):
        print(info)
        self.info.insertPlainText(info + "\n")

    def start_click(self):
        self.Print("即将开始制作数据集")
        self.dataset_name = self.name_dataset.text()
        self.norain_pth = self.dir_norain.text()
        self.rain_pth = self.dir_rain.text()
        self.frame_step = int(self.step.text())
        self.data_file = self.name_txt.text()
        self.char = self.name_char.text()
        flag = 1
        if self.dataset_name == "":
            flag = 0
            self.Print("错误:数据集名称不能为空")
        if self.rain_video == "":
            flag = 0
            self.Print("错误:未选择含雨视频")
        if self.norain_video == "":
            flag = 0
            self.Print("错误:未选择无雨视频")

        if self.create_directory1(self.dataset_name):
            flag = 0
            self.Print("错误:数据集名称已存在")

        if flag == 0:
            self.Print("\n请更正以上错误后再开始制作\n")

        if flag == 1:
            data_pth = "./" + self.dataset_name + "/"

            self.create_directory2(data_pth, self.rain_pth)
            self.create_directory2(data_pth, self.norain_pth)
            self.create_file(data_pth, self.data_file)

            rain_pth = data_pth + self.rain_pth + "/"
            norain_pth = data_pth + self.norain_pth + "/"
            data_file = data_pth + self.data_file
            self.stop = 0
            self.save_frames(self.rain_video, self.norain_video, rain_pth, norain_pth, data_file, self.frame_step, self.char)


if __name__ == '__main__':  # 运行主函数
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    Window = CreateDataset()
    Window.show()

    sys.exit(app.exec_())
