# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_Train.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1103, 780)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.la_title = QtWidgets.QLabel(self.centralwidget)
        self.la_title.setGeometry(QtCore.QRect(40, 20, 191, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.la_title.setFont(font)
        self.la_title.setObjectName("la_title")
        self.btn_choose = QtWidgets.QPushButton(self.centralwidget)
        self.btn_choose.setGeometry(QtCore.QRect(40, 60, 131, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.btn_choose.setFont(font)
        self.btn_choose.setObjectName("btn_choose")
        self.la_data_path = QtWidgets.QLabel(self.centralwidget)
        self.la_data_path.setGeometry(QtCore.QRect(190, 60, 741, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_data_path.setFont(font)
        self.la_data_path.setObjectName("la_data_path")
        self.la_pre_epoch = QtWidgets.QLabel(self.centralwidget)
        self.la_pre_epoch.setGeometry(QtCore.QRect(40, 100, 211, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_pre_epoch.setFont(font)
        self.la_pre_epoch.setObjectName("la_pre_epoch")
        self.la_pre_loss = QtWidgets.QLabel(self.centralwidget)
        self.la_pre_loss.setGeometry(QtCore.QRect(40, 130, 191, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_pre_loss.setFont(font)
        self.la_pre_loss.setObjectName("la_pre_loss")
        self.la_pre_psnr = QtWidgets.QLabel(self.centralwidget)
        self.la_pre_psnr.setGeometry(QtCore.QRect(40, 160, 191, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_pre_psnr.setFont(font)
        self.la_pre_psnr.setObjectName("la_pre_psnr")
        self.la_epoch = QtWidgets.QLabel(self.centralwidget)
        self.la_epoch.setGeometry(QtCore.QRect(40, 200, 131, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_epoch.setFont(font)
        self.la_epoch.setObjectName("la_epoch")
        self.line_epoch = QtWidgets.QLineEdit(self.centralwidget)
        self.line_epoch.setGeometry(QtCore.QRect(180, 196, 111, 26))
        self.line_epoch.setObjectName("line_epoch")
        self.line_lr = QtWidgets.QLineEdit(self.centralwidget)
        self.line_lr.setGeometry(QtCore.QRect(180, 228, 111, 26))
        self.line_lr.setObjectName("line_lr")
        self.la_lr = QtWidgets.QLabel(self.centralwidget)
        self.la_lr.setGeometry(QtCore.QRect(40, 232, 131, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_lr.setFont(font)
        self.la_lr.setObjectName("la_lr")
        self.la_gamma = QtWidgets.QLabel(self.centralwidget)
        self.la_gamma.setGeometry(QtCore.QRect(40, 264, 131, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_gamma.setFont(font)
        self.la_gamma.setObjectName("la_gamma")
        self.line_gamma = QtWidgets.QLineEdit(self.centralwidget)
        self.line_gamma.setGeometry(QtCore.QRect(180, 260, 111, 26))
        self.line_gamma.setObjectName("line_gamma")
        self.la_batch = QtWidgets.QLabel(self.centralwidget)
        self.la_batch.setGeometry(QtCore.QRect(40, 296, 131, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_batch.setFont(font)
        self.la_batch.setObjectName("la_batch")
        self.line_batch = QtWidgets.QLineEdit(self.centralwidget)
        self.line_batch.setGeometry(QtCore.QRect(180, 292, 111, 26))
        self.line_batch.setObjectName("line_batch")
        self.btn_start = QtWidgets.QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(40, 340, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.btn_start.setFont(font)
        self.btn_start.setObjectName("btn_start")
        self.btn_stop = QtWidgets.QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(160, 340, 91, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.btn_stop.setFont(font)
        self.btn_stop.setObjectName("btn_stop")
        self.text_info = QtWidgets.QTextEdit(self.centralwidget)
        self.text_info.setGeometry(QtCore.QRect(30, 590, 1021, 161))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.text_info.setFont(font)
        self.text_info.setObjectName("text_info")
        self.progress = QtWidgets.QProgressBar(self.centralwidget)
        self.progress.setGeometry(QtCore.QRect(230, 530, 251, 21))
        self.progress.setProperty("value", 24)
        self.progress.setObjectName("progress")
        self.la_cur_epoch = QtWidgets.QLabel(self.centralwidget)
        self.la_cur_epoch.setGeometry(QtCore.QRect(40, 530, 121, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_cur_epoch.setFont(font)
        self.la_cur_epoch.setObjectName("la_cur_epoch")
        self.la_cur_psnr = QtWidgets.QLabel(self.centralwidget)
        self.la_cur_psnr.setGeometry(QtCore.QRect(230, 560, 191, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_cur_psnr.setFont(font)
        self.la_cur_psnr.setObjectName("la_cur_psnr")
        self.la_cur_loss = QtWidgets.QLabel(self.centralwidget)
        self.la_cur_loss.setGeometry(QtCore.QRect(40, 560, 171, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_cur_loss.setFont(font)
        self.la_cur_loss.setObjectName("la_cur_loss")
        self.check_stop = QtWidgets.QCheckBox(self.centralwidget)
        self.check_stop.setGeometry(QtCore.QRect(163, 380, 91, 21))
        self.check_stop.setObjectName("check_stop")
        self.window = QtWidgets.QLabel(self.centralwidget)
        self.window.setGeometry(QtCore.QRect(310, 90, 741, 421))
        self.window.setObjectName("window")
        self.la_time = QtWidgets.QLabel(self.centralwidget)
        self.la_time.setGeometry(QtCore.QRect(500, 530, 191, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        self.la_time.setFont(font)
        self.la_time.setText("")
        self.la_time.setObjectName("la_time")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.la_title.setText(_translate("MainWindow", "去雨模型训练工具"))
        self.btn_choose.setText(_translate("MainWindow", "选择数据集目录"))
        self.la_data_path.setText(_translate("MainWindow", "未选择                                                                   |"))
        self.la_pre_epoch.setText(_translate("MainWindow", "当前已训练轮次:0"))
        self.la_pre_loss.setText(_translate("MainWindow", "Loss:"))
        self.la_pre_psnr.setText(_translate("MainWindow", "PSNR:"))
        self.la_epoch.setText(_translate("MainWindow", "Epochs："))
        self.line_epoch.setText(_translate("MainWindow", "10"))
        self.line_lr.setText(_translate("MainWindow", "0.0001"))
        self.la_lr.setText(_translate("MainWindow", "LearningRate："))
        self.la_gamma.setText(_translate("MainWindow", "Gamma："))
        self.line_gamma.setText(_translate("MainWindow", "0.99"))
        self.la_batch.setText(_translate("MainWindow", "BatchSize："))
        self.line_batch.setText(_translate("MainWindow", "4"))
        self.btn_start.setText(_translate("MainWindow", "开始训练"))
        self.btn_stop.setText(_translate("MainWindow", "立即结束"))
        self.text_info.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'黑体\',\'黑体\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:14pt;\"><br /></p></body></html>"))
        self.la_cur_epoch.setText(_translate("MainWindow", "Epoch:"))
        self.la_cur_psnr.setText(_translate("MainWindow", "PSNR:"))
        self.la_cur_loss.setText(_translate("MainWindow", "Loss:"))
        self.check_stop.setText(_translate("MainWindow", "提前结束"))
        self.window.setText(_translate("MainWindow", "|                                                                                           |"))
