import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    signalImg = pyqtSignal(np.ndarray)
    signalreplay = pyqtSignal(bool)
    def __init__(self, url_video):
        super().__init__()
        self.url_video = url_video
        # self.area1 = area1
        self.player = cv2.VideoCapture(self.url_video)
        self.threadActive = False

    # def get_video_from(self):
    #     return cv2.VideoCapture(self.url_video)
        # return cv2.VideoCapture('D:\Project_Atin\highway.mp4')

    def run(self):
        self.threadActive = True
        # self.cap.open(self.url_video)
        while self.threadActive:
            ret, frame = self.player.read()
            if not ret:
                self.player = cv2.VideoCapture(self.url_video)
                # self.signalreplay.emit(True)
                continue
            else:
                
                self.signalImg.emit(frame)
            self.msleep(30)
        self.player.release()

    def stop(self):
        self.threadActive = False


        

        
    