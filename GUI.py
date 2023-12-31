# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'atin.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition

from configUI import Ui_Form
import cv2
# import cvzone
from info import Ui_Info
from Capture import CaptureThread
from Inference import Inference

# fpsReader = cvzone.FPS()
class Ui_MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.list_info = []
        self.pause = False
        
        self.list_car_img = []
        self.list_color = []
        self.list_brand = []


        self.cell = 0
        self.count = 0
        self.show()
        self.reset = False
        for i in range(6):
            info = Ui_Info()
            # print(info)
            self.list_info.append(info)
            self.scrollAreaWidgetContents.layout().addWidget(info)
        
    def setupUi(self):
        # self.setObjectName()
        self.resize(1920, 1045)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setAutoFillBackground(True)
        self.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setStyleSheet("background-color: #0c5e94")
        self.centralwidget.setObjectName("centralwidget")
        self.count_label = QtWidgets.QLabel(self.centralwidget)
        self.count_label.setGeometry(QtCore.QRect(270, 440, 90, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.count_label.sizePolicy().hasHeightForWidth())
        self.count_label.setSizePolicy(sizePolicy)
        self.count_label.setMaximumSize(QtCore.QSize(100, 72))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.count_label.setFont(font)
        self.count_label.setStyleSheet("background-color: #aaa900;\n"
"")
        self.count_label.setObjectName("count_label")
        self.video_label = QtWidgets.QLabel(self.centralwidget)
        self.video_label.setGeometry(QtCore.QRect(400, 30, 1481, 701))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.video_label.sizePolicy().hasHeightForWidth())
        self.video_label.setSizePolicy(sizePolicy)
        self.video_label.setText("")
        self.video_label.setPixmap(QtGui.QPixmap("Image/Kokushibo_29.png"))
        self.video_label.setScaledContents(True)
        self.video_label.setObjectName("video_label")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(51, 25, 341, 391))
        self.layoutWidget.setMinimumSize(QtCore.QSize(341, 391))
        self.layoutWidget.setMaximumSize(QtCore.QSize(341, 391))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Logo_label = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Logo_label.sizePolicy().hasHeightForWidth())
        self.Logo_label.setSizePolicy(sizePolicy)
        self.Logo_label.setMinimumSize(QtCore.QSize(339, 191))
        self.Logo_label.setMaximumSize(QtCore.QSize(339, 191))
        self.Logo_label.setText("")
        self.Logo_label.setPixmap(QtGui.QPixmap("Image/Atin.png"))
        self.Logo_label.setScaledContents(True)
        self.Logo_label.setObjectName("Logo_label")
        self.verticalLayout_2.addWidget(self.Logo_label)
        self.widget = QtWidgets.QWidget(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(339, 191))
        self.widget.setMaximumSize(QtCore.QSize(339, 191))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.config_but = QtWidgets.QPushButton(self.widget)
        self.config_but.setEnabled(True)
        self.config_but.setMinimumSize(QtCore.QSize(317, 28))
        self.config_but.setMaximumSize(QtCore.QSize(317, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.config_but.setFont(font)
        self.config_but.setAutoFillBackground(False)
        self.config_but.setStyleSheet("border-radius: 5px;\n"
"background-color: #aaa900;\n"
"")
        
        self.video_label.setStyleSheet("border: 2px solid white;")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/CG.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.config_but.setIcon(icon)
        self.config_but.setObjectName("config_but")
        self.verticalLayout.addWidget(self.config_but)
        self.start_but = QtWidgets.QPushButton(self.widget)
        self.start_but.setMinimumSize(QtCore.QSize(317, 28))
        self.start_but.setMaximumSize(QtCore.QSize(317, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.start_but.setFont(font)
        self.start_but.setStyleSheet("border-radius: 5px;\n"
"background-color: #aaa900;\n"
"")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/start.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.start_but.setIcon(icon1)
        self.start_but.setObjectName("start_but")
        self.verticalLayout.addWidget(self.start_but)
        self.stop_but = QtWidgets.QPushButton(self.widget)
        self.stop_but.setMinimumSize(QtCore.QSize(317, 28))
        self.stop_but.setMaximumSize(QtCore.QSize(317, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.stop_but.setFont(font)
        self.stop_but.setStyleSheet("border-radius: 5px;\n"
"background-color: #aaa900;\n"
"")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stop_but.setIcon(icon2)
        self.stop_but.setObjectName("stop_but")
        self.verticalLayout.addWidget(self.stop_but)
        self.find_but = QtWidgets.QPushButton(self.widget)
        self.find_but.setMinimumSize(QtCore.QSize(317, 28))
        self.find_but.setMaximumSize(QtCore.QSize(317, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.find_but.setFont(font)
        self.find_but.setStyleSheet("border-radius: 5px;\n"
"background-color: #aaa900;\n"
"")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon/find.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.find_but.setIcon(icon3)
        self.find_but.setObjectName("find_but")
        self.verticalLayout.addWidget(self.find_but)
        self.verticalLayout_2.addWidget(self.widget)
        self.car_label = QtWidgets.QLabel(self.centralwidget)
        self.car_label.setGeometry(QtCore.QRect(60, 430, 321, 301))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.car_label.sizePolicy().hasHeightForWidth())
        self.car_label.setSizePolicy(sizePolicy)
        self.car_label.setStyleSheet("border-radius: 5px;")
        self.car_label.setText("")
        self.car_label.setPixmap(QtGui.QPixmap("Image/car.png"))
        self.car_label.setScaledContents(True)
        self.car_label.setObjectName("car_label")

        self.layout_main = QtWidgets.QGridLayout(self.layoutWidget)
        self.layout_main.setObjectName("layout_main")
        # self.layout_main.setGeometry(QtCore.QRect(70, 770, 1801, 201))
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(50, 770, 1841, 201))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1799, 199))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.H_Layout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.H_Layout.setObjectName("H_Layout")
        self.layout_main.addWidget(self.scrollArea, 1, 0, 1, 2)

        self.car_label.raise_()
        self.count_label.raise_()
        self.video_label.raise_()
        self.layoutWidget.raise_()
        self.scrollArea.raise_()
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        # QtCore.QMetaObject.connectSlotsByName(self)

        self.url_video = ''
        self.config_but.clicked.connect(self.config)
        self.start_but.clicked.connect(self.start_capture)
        self.stop_but.clicked.connect(self.stop_capture)
        
        
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.count_label.setText(_translate("MainWindow", "0"))
        self.config_but.setText(_translate("MainWindow", "Thiết lập cấu hình"))
        self.start_but.setText(_translate("MainWindow", "Bắt đầu"))
        self.stop_but.setText(_translate("MainWindow", "Kết thúc"))
        self.find_but.setText(_translate("MainWindow", "Tìm kiếm"))

        

    def config(self):
        self.configWindow = QMainWindow()
        self.configUI = Ui_Form()
        self.configUI.setupUi(self.configWindow)
        self.configWindow.show()

        self.configUI.url_signal.connect(self.get_url)

        
    # def reset_info(self):
    #     fo

        
    def get_url(self, url):
        
        if self.reset is True:
            while not self.Inference.queue.empty(): 
                self.Inference.queue.get()
            # self.Inference.stop()
            self.threadCapture.stop()
            self.list_info = []
            self.list_car_img=[]
            self.list_color = []
            layout = self.scrollAreaWidgetContents.layout()
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            for i in range(6):
                info = Ui_Info()
                # print(info)
                self.list_info.append(info)
                self.scrollAreaWidgetContents.layout().addWidget(info)
            self.count_label.setText("0")
        self.reset = True
        self.url_video = url
        print("Đã lấy được Url video: {}".format(self.url_video))
        video_Cap = cv2.VideoCapture(self.url_video)
        video_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video_Cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Tạo QImage từ khung hình
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap(q_image))
            url_area = self.url_video[:-3]+'txt'   
            with open(url_area, 'r') as file:
                content = file.read().split(',')
                self.area1 = [(int(content[0]), int(content[1])),(int(content[2]), int(content[3])),(int(content[4]), int(content[5])),(int(content[6]), int(content[7]))]
                self.line = (int(content[4]), int(content[5])),(int(content[6]), int(content[7]))
                # print(self.area1)

    def start_capture(self):
        
        _translate = QtCore.QCoreApplication.translate
        string =str(self.url_video)
        if not string:
            mess = QMessageBox()
            mess.setWindowTitle('cái đ gì???????')
            mess.setText('Chưa chọn video kìa???')
            mess.setIcon(QMessageBox.Warning)  #Critical, Warning, Information, Question
            mess.setStandardButtons(QMessageBox.Ok)
            mess.setDefaultButton(QMessageBox.Ok)
            x = mess.exec_()

        else:

            
            self.count_label.setText("0")
            self.threadCapture = CaptureThread(self.url_video)
            
            self.Inference = Inference(self.area1)

            
            self.threadCapture.signalImg.connect(self.get_frame)

            self.Inference.signal.connect(self.show_wedcam)
            self.Inference.update_count.connect(self.count_update)

            self.Inference.CAR_signal.connect(self.Display_Info)

            self.threadCapture.start()
            self.Inference.start()
            


    def Display_Info(self, img, color, brand):
        if int(self.count) != self.cell and self.cell < 6:
            self.cell = int(self.count)
            if self.cell>1:
                for i in range(1,self.cell):
                    inf_i = self.list_info[i]
                    inf_i.image_label.setPixmap(self.list_car_img[i-1])
                    inf_i.label_color.setText(self.list_color[i-1])
                    inf_i.label_brand.setText(self.list_brand[i-1])

            info = self.list_info[0]
            qt_img = self.convert_cv_qt(img)
            qt_img = QPixmap(qt_img)
            qt_img = qt_img.scaled(info.image_label.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            
            info.image_label.setPixmap(qt_img)
            info.label_color.setText(color)
            info.label_brand.setText(brand)
            self.list_color.insert(0,color)
            self.list_car_img.insert(0,qt_img)
            self.list_brand.insert(0, brand)

            
        elif int(self.count) != self.cell and self.cell >= 6:
            self.cell = int(self.count)
            for i in range(1,6):
                inf_i = self.list_info[i]
                inf_i.image_label.setPixmap(self.list_car_img[i-1])
                inf_i.label_color.setText(self.list_color[i-1])
                inf_i.label_brand.setText(self.list_brand[i-1])

            info = self.list_info[0]
            qt_img = self.convert_cv_qt(img)
            qt_img = QPixmap(qt_img)
            qt_img = qt_img.scaled(info.image_label.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            
            info.image_label.setPixmap(qt_img)
            info.label_color.setText(color)
            info.label_brand.setText(brand)
            self.list_color.insert(0,color)
            self.list_car_img.insert(0,qt_img)
            self.list_brand.insert(0,brand)
            if len(self.list_car_img) > 6:
                self.list_car_img = self.list_car_img[:-1]
                self.list_color = self.list_color[:-1]
                self.list_brand = self.list_brand[:-1]

            

    def count_update(self, count):     
        self.count = count
        self.count_label.setText("{}".format(self.count))
        

    def stop_capture(self):
        _translate = QtCore.QCoreApplication.translate
        string =str(self.url_video)
        if not string:
            mess = QMessageBox()
            mess.setWindowTitle('Stop cái đ gì???????')
            mess.setText('chọn video đi thằng ngu, chưa chọn vid thì stop cái đ gì')
            mess.setIcon(QMessageBox.Warning)  #Critical, Warning, Information, Question
            mess.setStandardButtons(QMessageBox.Ok)
            mess.setDefaultButton(QMessageBox.Ok)
            x = mess.exec_()
        else:
            if not self.pause:
                self.Inference.pause_stream()
                self.pause = True
            else:
                self.Inference.resume_stream()
                self.pause = False

    def get_frame(self, frame):
        self.Inference.queue.put(frame)

    def show_wedcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)      
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image =cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # fps, rgb_image = fpsReader.update(rgb_image,pos=(50,80),color=(0,255,0),scale=5,thickness=5)
        h,w,ch = rgb_image.shape
        bytes_per_line = ch * w 
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1080, 720, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p) 

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    
    sys.exit(app.exec_())
