# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
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


class Ui_MainWindow(QWidget):
    ##初始化UI界面
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("TaichiPose")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(110, 370, 221, 71))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.upload)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 370, 221, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.excute)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 90, 271, 241))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setAutoFillBackground(True)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(450, 90, 271, 241))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_2.setAutoFillBackground(True)
        ##将图片位置填充成白色
        palette = QPalette()
        palette.setColor(QPalette.Window,Qt.red)
        self.label.setPalette(palette)
        palette.setColor(QPalette.Window, Qt.blue)
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
    def upload(self):##上传图像，并检测图片尺寸，尺寸应小于
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.uploadimg=imgName
        print(imgName)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        img = demo_camera.cv_imread(self.uploadimg)
        if img.shape[0]>1000 or img.shape[1]>1000:
            self.ImageSizeError()
    def ImageSizeError(self):##图片过大弹窗
        QMessageBox.critical(self,"错误","图片过大，请重新上传",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
    def excute(self):##开线程，线程函数为work
        thread = Thread(target=self.work)
        thread.start()
    def work(self):##调用Pose函数处理图片并显示在窗口上
        self.pushButton_2.setDisabled(True)
        tic=time.time()
        canvas=demo_camera.Pose(self.uploadimg)
        for i in range(20):
            QApplication.processEvents()
        cv2.imwrite("result.png", canvas)
        jpg = QtGui.QPixmap("result.png").scaled(self.label.width(), self.label.height())
        self.label_2.setPixmap(jpg)
        toc=time.time()
        print("The process time is %.1fs"%(toc-tic))
        self.pushButton_2.setDisabled(False)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())