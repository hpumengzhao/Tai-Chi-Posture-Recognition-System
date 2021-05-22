# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
#1.2->1.3
#tasks:增加等待窗口和时间通知
#bugfixed：解决了大文件button2点击卡死的bug
import time
import sys
import cv2
import demo_camera
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QApplication
from threading import Thread
import multiprocessing

class Ui_MainWindow(QWidget):
    ##设置信号通道
    sinOut = pyqtSignal(str)
    sinOut2 = pyqtSignal(str)
    def setupUi(self, MainWindow):##初始化UI界面
        MainWindow.setObjectName("TaichiPose1.3")
        MainWindow.resize(800, 600)
        MainWindow.setFixedSize(800,600)
        ##队列槽函数connect，一旦对应队列接收到信号，就去执行对应的操作
        self.sinOut.connect(self.GetProcessTime)
        self.sinOut2.connect(self.ProcessingWindow)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(110, 370, 221, 71))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.upload)##上传图片

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 370, 221, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.excute)##开辟线程，处理图片

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 90, 271, 241))
        self.label.setText("Input Image")
        self.label.setObjectName("label")
        self.label.setAutoFillBackground(True)
        self.label.setAlignment(Qt.AlignCenter)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(450, 90, 271, 241))
        self.label_2.setText("OutPut Image")
        self.label_2.setObjectName("label_2")
        self.label_2.setAutoFillBackground(True)
        self.label_2.setAlignment(Qt.AlignCenter)
        ##将图片位置填充成白色
        palette = QPalette()
        palette.setColor(QPalette.Window,Qt.red)
        self.label.setPalette(palette)
        palette.setColor(QPalette.Window, Qt.yellow)
        self.label_2.setPalette(palette)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.uploadimg=""
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("TaichiPose", "TaichiPose"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_2.setText(_translate("MainWindow", "姿态识别"))
    def upload(self):##上传图像，并检测图片尺寸，尺寸应小于1000*1000
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if imgName=='':##关闭文件夹则pass（即文件为空）
            pass
        else:
            self.uploadimg=imgName
            jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
            img = demo_camera.cv_imread(self.uploadimg)
            if img.shape[0]>1000 or img.shape[1]>1000:
                self.pushButton_2.setDisabled(True)
                self.ImageSizeError()
            else:
                self.pushButton_2.setDisabled(False)
    def ImageSizeError(self):##图片过大弹窗
        QMessageBox.critical(self,"错误","图片过大，请重新上传",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
    def ImageEmptyError(self):##图片为空警告
        QMessageBox.information(self, "错误", "图片为空", QMessageBox.Ok,QMessageBox.Ok)
    def ProcessingWindow(self,rcv):
        if rcv=="end":##处理结束后接收到"end"信号，关闭处理窗口
            self.infoBox.close()
        else:##接收到开始处理信号"start"
            self.infoBox = QMessageBox()
            self.infoBox.setIcon(QMessageBox.Information)
            self.infoBox.setText("正在处理......")
            self.infoBox.setWindowTitle("Processing")
            self.infoBox.addButton(QMessageBox.Ok)
            self.infoBox.button(QMessageBox.Ok).hide()
            self.infoBox.exec_()

    def excute(self):##开线程，线程函数为work
        if self.uploadimg=='':
            self.ImageEmptyError()
        else:
            thread = Thread(target=self.work)
            thread.start()
    def GetProcessTime(self,rst): ##rst即为信号
        QMessageBox.information(self, "Success!", "处理时间: "+rst+"s")
    def work(self):##调用Pose函数处理图片并显示在窗口上
        self.sinOut2.emit("start") ##向信号队列sinOut2发送start信号
        self.pushButton_2.setDisabled(True)
        tic=time.time()
        canvas=demo_camera.Pose(self.uploadimg)
        cv2.imwrite("result.png", canvas)
        jpg = QtGui.QPixmap("result.png").scaled(self.label.width(), self.label.height())
        self.label_2.setPixmap(jpg)
        toc=time.time()
        print("The process time is %.1fs"%(toc-tic))
        self.pushButton_2.setDisabled(False)
        durationT = round((toc-tic),2)
        self.sinOut2.emit("end")###处理结束后向信号队列sinOut2发送end信号
        self.sinOut.emit(str(durationT))##向信号队列sinOut发送时间信号
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())