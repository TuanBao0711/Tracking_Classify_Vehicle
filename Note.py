from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from configUI import Ui_Form



def config(self):
    self.configWindow = QMainWindow()
    self.configUI = Ui_Form()
    self.configUI.setupUi(self.configWindow)
    self.configWindow.show()




from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject

def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.url_video = file_path
            url_lst = self.url_video.split('/')
            url_lst= url_lst[8:]
            self.url_video = '/'.join(url_lst)
            self.lineEdit_2.setText(self.url_video)
def OK(self):
        self.url_signal.emit(self.url_video)