# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'config.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget,QApplication, QMainWindow, QFileDialog, QMessageBox, QDesktopWidget , QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject

class Ui_Form(QWidget):
    reset_signal = pyqtSignal(str,bool)
    reset = False
    url_signal=pyqtSignal(str)
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1162, 797)
        Form.setAutoFillBackground(False)
        Form.setStyleSheet("background-color: #0c5e94")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(90, 60, 111, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.UrlButton = QtWidgets.QPushButton(Form)
        self.UrlButton.setGeometry(QtCore.QRect(860, 310, 111, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.UrlButton.sizePolicy().hasHeightForWidth())
        self.UrlButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.UrlButton.setFont(font)
        self.UrlButton.setAutoFillBackground(False)
        self.UrlButton.setStyleSheet("border-radius: 5px; background-color: #b4d9e3")
        self.UrlButton.setObjectName("UrlButton")
        self.checkBox = QtWidgets.QCheckBox(Form)
        self.checkBox.setGeometry(QtCore.QRect(870, 500, 111, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(80, 500, 741, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox.setFont(font)
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(90, 250, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.OK_but = QtWidgets.QPushButton(Form)
        self.OK_but.setGeometry(QtCore.QRect(80, 620, 111, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OK_but.sizePolicy().hasHeightForWidth())
        self.OK_but.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.OK_but.setFont(font)
        self.OK_but.setAutoFillBackground(False)
        self.OK_but.setStyleSheet("border-radius: 5px; background-color: #b4d9e3")
        self.OK_but.setObjectName("OK_but")
        self.Cancel_but = QtWidgets.QPushButton(Form)
        self.Cancel_but.setGeometry(QtCore.QRect(220, 620, 111, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Cancel_but.sizePolicy().hasHeightForWidth())
        self.Cancel_but.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Cancel_but.setFont(font)
        self.Cancel_but.setAutoFillBackground(False)
        self.Cancel_but.setStyleSheet("border-radius: 5px; background-color: #b4d9e3")
        self.Cancel_but.setObjectName("Cancel_but")
        self.LineEdit = QtWidgets.QLineEdit(Form)
        self.LineEdit.setGeometry(QtCore.QRect(80, 110, 741, 41))
        self.LineEdit.setObjectName("LineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form)
        self.lineEdit_2.setGeometry(QtCore.QRect(80, 310, 741, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


        self.UrlButton.clicked.connect(self.open_file)
        self.OK_but.clicked.connect(self.OK_config)
        self.Cancel_but.clicked.connect(self.Cancel_but.parent().close)
        # self.OK_but.clicked.connect(self.OK_but.parent().close)


        self.url_video = ''
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "CamID"))
        self.UrlButton.setText(_translate("Form", "Linkk"))
        self.checkBox.setText(_translate("Form", "Hiển thị"))
        self.comboBox.setItemText(0, _translate("Form", "Phát hiện biển số"))
        self.comboBox.setItemText(1, _translate("Form", "Something"))
        self.comboBox.setItemText(2, _translate("Form", "anything"))
        self.label_2.setText(_translate("Form", "RTPS"))
        self.OK_but.setText(_translate("Form", "Ok"))
        self.Cancel_but.setText(_translate("Form", "Cancel"))


    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.url_video = file_path
            url_lst = self.url_video.split('/')
            url_lst= url_lst[8:]
            self.url_video = '/'.join(url_lst)
            self.lineEdit_2.setText(self.url_video)
    def OK_config(self):
        
        if self.url_video:
            
            # self.reset_signal.emit(self.reset)
            # self.url_signal.emit(self.url_video, self.reset_signal)
            self.url_signal.emit(self.url_video)
            # print('Đã gửi tín hiệu:', self.url_video)
            # print(self.Cancel_but.parent())

            self.reset = True
            self.OK_but.parent().close()
        else:
            mess = QMessageBox()
            mess.setWindowTitle('Lỗi')
            mess.setText('Chưa lấy video kìa thằng ngu!!!')
            mess.setIcon(QMessageBox.Warning)  #Critical, Warning, Information, Question
            mess.setStandardButtons(QMessageBox.Ok)
            mess.setDefaultButton(QMessageBox.Ok)
            x = mess.exec_()
        self.reset = True
        
            


    def close_widget(self):
        # Đóng widget
        self.close()

    def Can_Config(self):
        pass
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
