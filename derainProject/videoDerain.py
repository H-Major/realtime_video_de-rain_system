from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QTime
from PyQt5 import QtCore
import sys
import os
import cv2
import numpy as np
from datetime import datetime

from videoDerainUi import Ui_MainWindow  # Qt的UI文件
# from rainClassifyNet import rainClassify  # 分类所需函数 --hidden-import rainClassifyNet.py
from derainNet import *  # 去雨所需函数

''' 
在Qt Designer中修改UI界面后，需要在终端中输入以下命令来更新UI界面的.py文件
python -m PyQt5.uic.pyuic derainUI.ui -o videoDerainUi.py

打包要使用以下命令，其中要--hidden-import+所有直接或间接依赖项
pyinstaller -w -D -i ./ui_file/img.ico videoDerain.py --hidden-import derainNet.py  --hidden-import videoDerainUi.py

待开发的功能如下：
录像功能（设置保存路径）、原分辨率显示窗口、混合模式、查看所有设置内容

'''

# 运算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 类别列表
class_List = ['无雨', '雨滴', '雨丝', '雪']
# 类别编号
no_rain = 0
point_rain_class = 1
line_rain_class = 2
snow_class = 3
# 存储位次编号
line_model_file_pth_num = 0
point_model_file_pth_num = 1
snow_model_file_pth_num = 2
# 模型路径存储文件地址
model_set_file_path = "settings_file/model_pth_setting.txt"


def get_current_time_string():
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    return current_time_str


def save_image(image, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, name)
    cv2.imwrite(full_path, image)
    print("save image " + full_path)


# 加载模型
def load_model(model_path):
    model = DerainModel()
    stadic = torch.load(model_path)
    model.load_state_dict(stadic)
    model = model.to(device)
    model.eval()
    return model


# 读取文件 按行返回列表
def read_txt_file(pth):
    lines = []
    with open(pth, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    file_to_read.close()
    return lines


# 获取当前时间（毫秒）
def get_current_time_milliseconds():
    current_time = QTime.currentTime()
    return current_time.msecsSinceStartOfDay()


def read_cap(cap_num):
    cap = cv2.VideoCapture(cap_num)
    ret, frame = cap.read()
    cap.release()
    return frame


# 写入文件 将列表分行写入
def write_txt_file(pth, pth_list):
    with open(pth, 'w') as file_to_write:
        for line in pth_list:
            file_to_write.write(line + '\n')
    file_to_write.close()


# 图片格式转换
def cvimg_to_qtimg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qtimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    return qtimg


class VideoDerainWindow(QMainWindow, Ui_MainWindow):
    # 构造函数
    def __init__(self):
        super(VideoDerainWindow, self).__init__()
        self.setupUi(self)

        print("构造函数")

        # 去雨视频窗口最大尺寸限制
        self.window_w = self.label_window.width()
        self.window_h = self.label_window.height()

        # 工作目录
        self.current_work_dir = str(os.path.dirname(__file__)).replace('\\', '/') + '/'
        print("当前工作目录：" + self.current_work_dir)

        # 默认去雨模型文件路径
        self.pre_line_model_path = "trained_model/line.pth"
        self.pre_point_model_path = "trained_model/point.pth"

        # 背景图片
        self.background_pix = QPixmap('ui_file/background.jpg')

        # 去雨模型文件路径
        self.line_model_path = read_txt_file(model_set_file_path)[line_model_file_pth_num]
        self.point_model_path = read_txt_file(model_set_file_path)[point_model_file_pth_num]
        self.snow_model_path = read_txt_file(model_set_file_path)[snow_model_file_pth_num]

        # 模型设置
        self.model_list = [0, 0, 0, 0]
        self.model_list[line_rain_class] = load_model(self.line_model_path)
        self.model_list[point_rain_class] = load_model(self.point_model_path)
        self.model_list[snow_class] = load_model(self.snow_model_path)

        # 暂停设置
        self.pause = 1

        # 摄像头
        self.capNum = 0

        # 摄像头是否正常
        self.capOK = 0

        # 分辨率
        self.W = 1920
        self.H = 1080

        # 去雨精度
        self.precision = 0
        self.size_list = [[512, 512], [1536, 864]]
        self.precision_list = self.size_list[self.precision]

        # 是否使用预设的分辨率
        self.usePreSet = 1

        # 去雨类别设置
        self.classes = line_rain_class

        # 视图设置
        self.view_set = 0

        # 文件去雨路径
        self.video_path = ""
        self.image_path = ""

        # 是否从文件读取
        self.useVideoFile = 0
        self.useImageFile = 0

        print("信号设置")
        # 信号设置
        # 复选框设置
        self.check_wh.clicked.connect(self.check_wh_clicked)  # 分辨率checkbox设置
        self.check_wxh.clicked.connect(self.check_wxh_clicked)  # 分辨率checkbox设置
        # 按钮设置
        self.btn_start.clicked.connect(self.start_pause_clicked)  # 开始暂停视频去雨
        self.btn_save.clicked.connect(self.save_and_apply)  # 保存并应用
        self.btn_close.clicked.connect(self.close_path_derain)  # 关闭视频文件
        self.btn_imagederain.clicked.connect(self.image_path_derain_func)  # 单图像去雨函数
        self.btn_captest.clicked.connect(self.test_all_cap)  # 检测所有摄像头
        # 菜单设置
        self.action_line.triggered.connect(self.set_line_model)  # 模型设置菜单
        self.action_point.triggered.connect(self.set_point_model)
        self.action_snow.triggered.connect(self.set_snow_model)
        self.action_reset.triggered.connect(self.reset_model_file)
        self.action_one.triggered.connect(self.view_one)  # 视图设置菜单
        self.action_double.triggered.connect(self.view_double)
        self.action_video.triggered.connect(self.video_path_derain)  # 文件读取菜单设置
        self.action_image.triggered.connect(self.image_path_derain)
        self.action_showfile.triggered.connect(self.show_path_info)

        # 窗口设置
        self.setWindowTitle('视频去雨')  # 窗口名称
        self.setWindowIcon(QIcon("ui_file/img.ico"))
        self.set_background()  # 设置背景
        self.check_wxh.setChecked(True)  # 设置选中
        self.rbtn_line.setChecked(True)  # 设置选中
        self.label_window.setScaledContents(True)  # 设置label背景自适应
        self.label_window2.setScaledContents(True)
        self.label_window3.setScaledContents(True)
        self.label_window.setText("")  # 设置label文字为空
        self.label_window2.setText("")
        self.label_window2.hide()
        self.label_window3.setText("")
        self.label_window3.hide()
        self.action_one.setCheckable(True)  # 设置复选框
        self.action_double.setCheckable(True)
        self.action_one.setChecked(True)  # 设置选中
        self.set_window_size()  # 设置去雨窗口尺寸
        self.btn_close.hide()  # 隐藏按钮
        self.btn_imagederain.hide()  # 隐藏按钮
        self.show_in_one_window(cv2.imread("ui_file/start_image.png"))
        self.show_in_double_window(cv2.imread("ui_file/start_image.png"), cv2.imread("ui_file/start_image.png"))
        # 菜单名称设置
        self.action_line.setText("雨丝模型")
        self.action_point.setText("雨滴模型")
        self.action_reset.setText("恢复预设")
        self.action_one.setText("单视图")
        self.action_double.setText("对照视图")
        self.action_origin.setText("原分辨率视图")
        self.action_video.setText("读取视频文件去雨")
        self.action_image.setText("读取图片文件去雨")
        self.action_showfile.setText("显示文件路径")

        # 测试摄像头
        if not self.test_cap(self.capNum):
            self.show_info("[错误] 当前设备无摄像头，不能从摄像头获取视频，请选择读取本地文件进行去雨")
        else:
            self.test_all_cap()

    # 保存并应用
    def save_and_apply(self):
        # 如果正在播放，则暂停
        flag = 0
        if self.pause == 0:
            self.start_pause_clicked()
            flag = 1
        # 设置分辨率
        if self.useVideoFile == 0 and self.useImageFile == 0:  # 从文件读取则不再设置分辨率
            if self.usePreSet == 0:  # 不适用预设
                self.W = int(self.line_w.text())
                self.H = int(self.line_h.text())
            else:  # 使用预设
                self.use_pre_set_pixel()
        # 设置去雨精度
        self.precision = self.box_precision.currentIndex()
        self.precision_list = self.size_list[self.precision]
        # 设置去雨窗口尺寸
        self.set_window_size()
        # 设置摄像头编号
        if self.capNum != self.box_cap.currentIndex():
            if self.test_cap(self.box_cap.currentIndex()) == 0:  # 摄像头检测异常
                QMessageBox.warning(self, "错误", str(self.box_cap.currentIndex()) + "号摄像头异常，已自动切换为上次使用的" + str(self.capNum) + "号摄像头",
                                    QMessageBox.Ok, QMessageBox.Ok)
                self.box_cap.setCurrentIndex(self.capNum)  # 恢复上次选中的摄像头
            else:
                self.capNum = self.box_cap.currentIndex()
        # 设置去雨类别
        self.set_class()

        # 输出信息
        info_str = "[保存成功] 分辨率:" + str(self.W) + 'x' + str(self.H) + " 摄像头编号:" + str(self.capNum) + " 去雨类别:" + class_List[self.classes] + " "
        print(info_str)
        self.show_info(info_str)
        # 恢复播放
        if flag:
            self.start_pause_clicked()
            self.show_in_bar("设置已保存并应用")
        else:
            self.show_image()
            self.show_in_bar("设置已保存")

    # 视频去雨窗口 qt
    def video_derain_qt(self):
        if self.useVideoFile == 1:
            cap = cv2.VideoCapture(self.video_path)  # 读视频文件对象
        else:
            cap = cv2.VideoCapture(self.capNum)  # 读取摄像头对象
            cap.set(3, self.W)
            cap.set(4, self.H)
        i = 1
        ret, frame = cap.read()
        pre_time = get_current_time_milliseconds()
        print(frame.shape[:2])
        while ret:
            # 视频去雨
            derain_class = self.image_derain(frame)
            app.processEvents()
            ret, frame = cap.read()
            # 计算实时帧率
            cur_time = get_current_time_milliseconds()
            dert_time = cur_time - pre_time
            frame_rate = int(1000 / dert_time)
            pre_time = cur_time
            # 信息显示
            print("[处理帧信息] 当前帧数统计:", i)
            info = ("[处理帧信息]当前帧数统计:" + str(i) + "  去雨类别:" + class_List[derain_class] + "  输入分辨率:" +
                    str(int(self.W)) + "x" + str(int(self.H)) + "\n[实时帧率] " + str(frame_rate) + "\n")
            self.text_info.clear()
            self.text_info.insertPlainText(info)
            i = i + 1
            # 结束判断
            if self.pause == 1:
                cap.release()
                break

    # 单图像去雨功能函数
    def image_derain(self, frame):
        classified_class = self.classes
        image = frame
        '''
        # 分类
        if self.classes != auto_rain_class:
            # 不自动识别
            image = self.deRain(image, self.classes)
        else:
            # 自动识别
            classified_class = rainClassify(image)
            if classified_class != no_rain:
                # 有雨
                image = self.deRain(image, classified_class)
            # 无雨 time_str + 
        '''
        # 不自动识别
        image = self.deRain(image, self.classes)
        if self.useImageFile == 1:
            time_str = get_current_time_string()
            save_image(frame, "save_images/", time_str + "_rain.jpg")
            save_image(image, "save_images/", time_str + "_derain.jpg")
        # 降噪
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # 图像显示
        if self.view_set == 0:  # 单视图
            self.show_in_one_window(image)
        elif self.view_set == 1:  # 对照视图
            self.show_in_double_window(frame, image)
        return classified_class

    # 在window中显示图像
    def show_in_one_window(self, image):
        image = cvimg_to_qtimg(image)
        pixmap = QPixmap.fromImage(image)
        self.label_window.setPixmap(pixmap)

    def show_in_double_window(self, frame, image):
        image = cvimg_to_qtimg(image)
        pixmap = QPixmap.fromImage(image)
        frame = cvimg_to_qtimg(frame)
        pixmap2 = QPixmap.fromImage(frame)
        self.label_window3.setPixmap(pixmap)
        self.label_window2.setPixmap(pixmap2)

    # 去雨函数
    def deRain(self, img, classes):  # 输入图片和去雨模型
        print("[deRain]1")
        h, w = img.shape[:2]
        img2 = cv2.resize(img, (self.precision_list[0], self.precision_list[1]))
        print("[deRain]2")
        # img2 = cv2.resize(img, (1024, 1024))
        # img2 = cv2.resize(img, (512, 512))
        # img2 = cv2.resize(img, (h - (h % 16), w - (w % 16)))
        img2 = (img2 / 255.0).astype('float32')
        inputs = np.transpose(img2, (2, 0, 1))
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.to(device)
        print("[deRain]3")
        out = (self.model_list[classes])(inputs)
        print("[deRain]4")
        out = out.cpu().detach().numpy()
        out = (out * 255)
        out = np.clip(out, 0, 255).astype('uint8')
        out = np.squeeze(out)
        out = np.transpose(out, (1, 2, 0))
        out = cv2.resize(out, (w, h))
        return out

    # 设置背景图
    def set_background(self):  # 设置背景图片
        palette1 = QPalette()
        background = self.background_pix.scaled(self.width(), self.height())
        palette1.setBrush(self.backgroundRole(), QBrush(background))  # 设置背景图片
        self.setPalette(palette1)

    # 播放/暂停
    def start_pause_clicked(self):
        if (not self.useImageFile) and (not self.useVideoFile) and (not self.capOK):
            self.show_in_bar("摄像头异常")
        else:
            if self.pause == 1:
                self.set_pause()
                self.show_in_bar("去雨画面正在播放")
                app.processEvents()
                self.video_derain_qt()
            else:
                self.show_in_bar("去雨画面已暂停播放")
                self.set_pause()

    # 设置去雨窗口尺寸
    def set_window_size(self):
        # 大窗口设置
        if self.W / self.H >= self.window_w / self.window_h:
            self.label_window.setFixedSize(self.window_w, int(self.H * self.window_w / self.W))
        else:
            self.label_window.setFixedSize(int(self.W * self.window_h / self.H), self.window_h)
        # 小窗口设置
        mid_size = 20
        max_width = int((self.window_w - mid_size) / 2)
        max_height = self.window_h
        if self.W / self.H >= max_width / max_height:
            self.label_window2.setFixedSize(max_width, int(self.H * max_width / self.W))
            self.label_window3.setFixedSize(max_width, int(self.H * max_width / self.W))
        else:
            self.label_window2.setFixedSize(int(max_height * self.W / self.H), max_height)
            self.label_window3.setFixedSize(int(max_height * self.W / self.H), max_height)

    # 使用预设分辨率
    def use_pre_set_pixel(self):
        wh_str = self.box_wxh.currentText()
        wh_list = wh_str.split('x')
        self.W = int(wh_list[0])
        self.H = int(wh_list[1])

    # 设置暂停
    def set_pause(self):
        if self.pause == 0:
            self.pause = int(1)
            self.btn_start.setText("播放")
        else:
            self.pause = int(0)
            self.btn_start.setText("停止")

    # 设置点击checkbox_wh
    def check_wxh_clicked(self):
        if self.check_wh.isChecked():
            self.check_wh.setChecked(False)
            self.check_wxh.setChecked(True)
            self.usePreSet = 1
        else:
            self.check_wxh.setChecked(False)
            self.check_wh.setChecked(True)
            self.usePreSet = 0
        print("self.usePreSet:", self.usePreSet)

    # 设置点击checkbox_wh
    def check_wh_clicked(self):
        if self.check_wxh.isChecked():
            self.check_wxh.setChecked(False)
            self.check_wh.setChecked(True)
            self.usePreSet = 0
        else:
            self.check_wh.setChecked(False)
            self.check_wxh.setChecked(True)
            self.usePreSet = 1
        print("self.usePreSet:", self.usePreSet)

    # 设置去雨类别
    def set_class(self):
        if self.rbtn_line.isChecked():
            self.classes = line_rain_class
        else:
            if self.rbtn_point.isChecked():
                self.classes = point_rain_class
            else:
                if self.rbtn_snow.isChecked():
                    self.classes = snow_class

    # 展示第一帧
    def show_image(self):
        if self.useVideoFile == 1:  # 读取文件
            cap = cv2.VideoCapture(self.video_path)  # 读视频文件对象
            self.H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.set_window_size()
        elif self.useImageFile == 1:
            cap = cv2.VideoCapture(self.image_path)  # 读图片文件对象
            self.H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.set_window_size()
        else:  # 读取摄像头
            if self.capOK == 0:  # 如果摄像头异常则退出函数
                return 0
            cap = cv2.VideoCapture(self.capNum)  # 生成读取摄像头对象
            cap.set(3, self.W)
            cap.set(4, self.H)
        ret, frame = cap.read()
        if ret:
            image = cvimg_to_qtimg(frame)
            pixmap = QPixmap.fromImage(image)
            self.label_window.setScaledContents(True)
            self.label_window.setPixmap(pixmap)
            self.label_window2.setPixmap(pixmap)
            self.label_window3.setPixmap(pixmap)
            app.processEvents()
        cap.release()

    # 设置模型文件的路径
    def set_line_model(self):
        if self.pause == 0:
            self.show_in_bar("请先停止运行")
        else:
            print("Setting Line model")
            pth = self.set_model_pth(line_model_file_pth_num)
            if pth:
                print("Chosen file:", pth)
                self.line_model_path = pth
                self.model_list[line_rain_class] = load_model(pth)
                self.show_info("已更改Line模型路径:" + pth)
                self.show_in_bar("去除雨丝模型路径保存成功")

    def set_point_model(self):
        if self.pause == 0:
            self.show_in_bar("请先停止运行")
        else:
            print("Setting Point model")
            pth = self.set_model_pth(point_model_file_pth_num)
            if pth:
                print("Chosen file:", pth)
                self.point_model_path = pth
                self.model_list[point_rain_class] = load_model(pth)
                self.show_info("已更改Point模型路径:" + pth)
                self.show_in_bar("去除雨滴模型路径保存成功")

    def set_snow_model(self):
        if self.pause == 0:
            self.show_in_bar("请先停止运行")
        else:
            print("Setting Snow model")
            pth = self.set_model_pth(snow_model_file_pth_num)
            if pth:
                print("Chosen file:", pth)
                self.snow_model_path = pth
                self.model_list[snow_class] = load_model(pth)
                self.show_info("已更改Snow模型路径:" + pth)
                self.show_in_bar("去雪模型路径保存成功")

    # 设置模型路径
    def set_model_pth(self, model_pth_num):
        pth, file = QFileDialog.getOpenFileName(None, "选择文件", "", "Python Files (*.pth);;All Files (*)")
        if pth:
            pth = self.to_relative_path(pth)
            pth_list = read_txt_file(model_set_file_path)
            pth_list[model_pth_num] = pth
            write_txt_file(model_set_file_path, pth_list)
        return pth

    # 转换为相对路径
    def to_relative_path(self, path):
        print(self.current_work_dir)
        if self.current_work_dir in path:
            path = path.replace(self.current_work_dir, "")
            print("已将绝对路径转换为相对路径：" + path)
        return path

    # 恢复预设模型路径
    def reset_model_file(self):
        print("Reset model")
        pth_list = [self.pre_line_model_path, self.pre_point_model_path]
        write_txt_file(model_set_file_path, pth_list)
        self.show_info("已恢复预设模型文件路径")
        self.show_in_bar("已恢复预设去雨模型文件路径,重启生效")

    # 输出信息
    def show_info(self, info):
        self.text_info.clear()
        self.text_info.insertPlainText(info)

    # 根据窗口大小调整尺寸
    def set_item(self):
        # 设置背景图片
        self.set_background()
        # 显示图像label
        self.window_w = self.width() - self.label_window.x() - 40
        self.window_h = self.height() - self.label_window.y() - 190
        self.set_window_size()
        self.label_window2.move(self.label_window.x(), self.label_window.y())
        self.label_window3.move(self.label_window.x() + self.label_window2.width() + 20, self.label_window.y())
        # 输出信息窗口
        self.text_info.setFixedSize(self.window_w, 70)
        self.text_info.move(270, self.height() - 130)
        self.label_info.move(270, self.height() - 160)
        # 按钮位置
        self.btn_save.move(20, self.height() - 130)
        self.btn_start.move(20, self.height() - 90)
        self.btn_close.move(130, self.height() - 90)
        self.btn_imagederain.move(20, self.height() - 90)

    # 重写调整窗口大小函数
    def resizeEvent(self, event):
        self.set_item()

    def dehaze_rate_changed(self):
        self.show_in_bar(" 去雾率 " + str(int(self.slider_dehaze.value() / self.slider_dehaze.maximum() * 100)) + "%")

    # 设置视图
    def view_one(self):
        flag = 0
        if self.pause == 0:
            self.start_pause_clicked()
            flag = 1
        self.view_set = 0
        self.label_window.show()
        self.label_window2.hide()
        self.label_window3.hide()
        self.action_one.setChecked(True)
        self.action_double.setChecked(False)
        if flag:
            self.start_pause_clicked()
        else:
            self.show_image()
        self.show_in_bar("已切换为单视图")

    def view_double(self):
        flag = 0
        if self.pause == 0:
            self.start_pause_clicked()
            flag = 1
        self.view_set = 1
        self.label_window.hide()
        self.label_window2.show()
        self.label_window3.show()
        self.action_one.setChecked(False)
        self.action_double.setChecked(True)
        if flag:
            self.start_pause_clicked()
        else:
            self.show_image()
        self.show_in_bar("已切换为双视图")

    # 状态栏显示信息
    def show_in_bar(self, info):
        self.statusbar.showMessage(info, 3000)

    # 读取视频文件去雨
    def video_path_derain(self):
        if self.pause == 0:
            self.start_pause_clicked()
        pth, file = QFileDialog.getOpenFileName()
        if pth:
            self.video_path = pth
            self.useVideoFile = 1
            self.btn_close.show()
            self.btn_start.show()
            self.btn_imagederain.hide()
            self.show_image()
            self.show_in_bar("已读取视频文件")

    # 关闭文件
    def close_path_derain(self):
        if self.pause == 0:
            self.start_pause_clicked()
        self.useVideoFile = 0
        self.useImageFile = 0
        self.btn_close.hide()
        self.btn_start.show()
        self.btn_imagederain.hide()
        self.save_and_apply()
        self.show_image()
        self.show_in_bar("已关闭文件")

    # 读取图片去雨
    def image_path_derain(self):
        print("图像去雨")
        if self.pause == 0:
            self.start_pause_clicked()
        if self.useVideoFile == 1:
            self.close_path_derain()
        print("图像去雨3")
        pth, file = QFileDialog.getOpenFileName()
        if pth:
            self.useImageFile = 1
            self.image_path = pth
            self.btn_close.show()
            self.btn_start.hide()
            self.btn_imagederain.show()
            self.show_image()
            self.show_in_bar("已读取图像文件")

    # 图像文件去雨
    def image_path_derain_func(self):
        self.show_in_bar("正在图像去雨")
        cap = cv2.VideoCapture(self.image_path)  # 读视频文件对象
        ret, frame = cap.read()
        self.image_derain(frame)
        self.show_in_bar("图像去雨成功")

    # 检测摄像头是否正常
    def test_cap(self, cap_num):
        self.capOK = 0
        cap = cv2.VideoCapture(cap_num)
        if cap.isOpened():
            self.capOK = 1
        cap.release()
        return self.capOK

    # 检测全部摄像头
    def test_all_cap(self):
        i = 0
        while True:
            if self.test_cap(i):
                i += 1
            else:
                break
        self.show_info("[摄像头检测] 共检测到" + str(i) + "个摄像头")
        self.show_in_bar("共检测到" + str(i) + "个摄像头")

    # 显示模型路径信息
    def show_path_info(self):
        info = "去雨丝模型路径：" + self.line_model_path + '\n' + "去雨滴模型路径：" + self.point_model_path
        self.show_info(info)


if __name__ == '__main__':  # 运行主函数
    QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    Window = VideoDerainWindow()
    Window.show()

    sys.exit(app.exec_())
