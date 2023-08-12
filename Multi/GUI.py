import sys
from time import time, sleep
import cv2
import numpy as np 
import torch


from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from PyQt5.QtGui import QPixmap, QFont 
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy

from tracker import *
from Capture import CaptureThread
from Inference import Inference


import cv2



class Ui_MainWindow(object):
    # url_video = ''
    url_video = ''
    area1 = []
    pause = False
    frame = None
    def setupUi(self, MainWindow):
        screen_size = QDesktopWidget().screenGeometry(-1)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 900)
        
        MainWindow.setGeometry(0, 50, screen_size.width(), screen_size.height()-115)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        


        self.pushButton_Path = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Path.setGeometry(QtCore.QRect(60, 210, 170, 37))
        self.pushButton_Path.setObjectName("pushButton_Path")
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(60, 360, 170, 37))
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_Stop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Stop.setGeometry(QtCore.QRect(60, 420, 170, 37))
        self.pushButton_Stop.setObjectName("pushButton_Path")

        self.label_video = QtWidgets.QLabel(self.centralwidget)
        # self.label_video.setGeometry(QtCore.QRect(70, 50, 1280, 720))
        self.label_video.setObjectName("label_video")
        # self.label_video.setPixmap(QPixmap('D:/Project_Atin/test/messi.jpg'))
        
        self.label_video.setScaledContents(True)
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        self.label_result.setGeometry(QtCore.QRect(90, 670, 171, 131))
        self.label_result.setObjectName("label_result")
        font = QFont('Arial', 12)
        self.label_result.setFont(font)
       
        self.parent_layout = QtWidgets.QHBoxLayout()

        self.layout = QVBoxLayout()
        # self.layout.setGeometry(QtCore.QRect(10, 100, 250, 250))
        self.layout.addWidget(self.pushButton_Path)
        self.pushButton_Path.setFixedWidth(200)
        self.layout.addWidget(self.pushButton_start)
        self.pushButton_start.setFixedWidth(200)
        self.layout.addWidget(self.pushButton_Stop)
        self.pushButton_Stop.setFixedWidth(200)
        self.layout.addWidget(self.label_result)

        
        

        self.layout_vid = QVBoxLayout()
        self.layout_vid.addWidget(self.label_video)
        # self.layout_vid.setStyleSheet("border: 10px solid black")
       

        self.parent_layout.addLayout(self.layout)
        self.layout_vid.setGeometry(QtCore.QRect(250, 50,screen_size.width()-280, screen_size.height()-140))
        
        self.centralwidget.setLayout(self.parent_layout)

        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.pushButton_Path.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButton_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pushButton_Stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        

        self.label_video.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.label_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.pushButton_start.clicked.connect(self.start_capture)
        self.pushButton_Path.clicked.connect(self.open_file) 
        self.pushButton_Stop.clicked.connect(self.stop_capture)


        
        self.url_video= ''


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Path.setText(_translate("MainWindow", "Chọn file"))
        self.pushButton_start.setText(_translate("MainWindow", "Start"))
        self.pushButton_Stop.setText(_translate("MainWindow", "Stop"))

        self.label_result.setText(_translate("MainWindow", "Số phương tiện: 0"))


    def closeEvent(self, event):
        self.thread[1].stop_stream()

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
            # self.thread[1].stop()

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
            self.label_result.setText(_translate("MainWindow", "Số phương tiện: 0"))
            self.threadCapture = CaptureThread(self.url_video)
            self.Inference = Inference(self.area1)
            # self.threadCapture.get_video_from()
            
            self.threadCapture.signalImg.connect(self.get_frame)
            # self.threadCapture.signalreplay.connect(self.replay)
            self.Inference.signal.connect(self.show_wedcam)
            self.Inference.update_count.connect(self.count_update)

            self.threadCapture.start()
            self.Inference.start()
            

    def get_frame(self, frame):
        self.Inference.queue.put(frame)

    def show_wedcam(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.label_video.setPixmap(qt_img)
    
    def replay(self, count):
        self.label_result.setText("Số phương tiện: {}".format(count))

    def count_update(self, count):
       
        self.label_result.setText("Số phương tiện: {}".format(count))



    def convert_cv_qt(self, cv_img):
        rgb_image =cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_image.shape
        bytes_per_line = ch * w 
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1080, 720, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.url_video = file_path
            url_area = file_path[:-3]+'txt'
           
            with open(url_area, 'r') as file:
                content = file.read().split(',')
                self.area1 = [(int(content[0]), int(content[1])),(int(content[2]), int(content[3])),(int(content[4]), int(content[5])),(int(content[6]), int(content[7]))]

                print(self.area1)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

