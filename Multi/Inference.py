import cv2
import numpy as np
import time
from  queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from tracker import *
import torch
import sys, os
from yolov5.models.common import *


# Sau đó, bạn có thể import module utils trong Inference.py
# 


# from yolov5.models.experimental import attempt_load
# from yolov5.utils.general import torch_utils
# from yolov5.models.yolo import Model

class Inference(QThread):
    signal = pyqtSignal(np.ndarray)
    update_count = pyqtSignal(str)

    # area1 = [(1120,1000),(1128,1162),(1336,1180),(1284,1019)]
    area2 = set()
    count = 0
    tracker = Tracker()
    def __init__(self, area1):
        super().__init__()
        self.frame = None
        self.threadActive = False

        self.device = None
        self.out_file = None
        self.classes = None
        self.model = None
        
        # self.url_video = url_video
        self.area1 = area1

        self.mutex = QMutex()
        self.is_paused = False
        self.is_running = True
        self.wait_cond = QWaitCondition()

    # def getFrame(self, frame):
        # self.frame = frame

        self.queue = Queue()


    def load(weights, device='cpu', half=False, verbose=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoShape(DetectMultiBackend(weights, device=device, fuse=True, fp16=half), verbose=verbose)
        return model


    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # model = self.load("yolov5s.onnx", "cpu")
        return model

    

    def run(self):
        print('Bắt đầu Inference!')
        self.threadActive = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = 'labeled_video.avi'
        
        while self.threadActive:
            self.frame = self.queue.get()
            if self.frame is not None:

                self.mutex.lock()
                while self.is_paused:
                    self.wait_cond.wait(self.mutex)
              
                # img_tensor = torch.tensor(np.array(self.frame)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                # results=self.model(img_tensor)
                # print(type(results))
                results=self.model(self.frame)
                lst = []

                for index, rows in results.pandas().xyxy[0].iterrows():
                    x = int(rows[0])
                    y = int(rows[1])
                    x1 = int(rows[2])
                    y1 = int(rows[3])
                    b = str(rows['name'])
                    lst.append([x,y,x1,y1])
                    # cv2.rectangle(frame,(x,y),(x1,y1), (0,0,255),2)
                idx_bbox = self.tracker.update(lst)
                for bbox in idx_bbox:
                    x2,y2,x3,y3,id = bbox
                    cv2.rectangle(self.frame, (x2,y2), (x3,y3), (0,0,255),2)
                    cv2.putText(self.frame, str(id),(x2,y2), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
                    cv2.circle(self.frame, ((x2+x3)//2,(y2+y3)//2),4,(0,255,0),-1)
                    result1 = cv2.pointPolygonTest(np.array(self.area1,np.int32), (((x2+x3)//2,(y2+y3)//2)),False)        
                    if result1 >0:
                        self.area2.add(id)


                cv2.polylines(self.frame,[np.array(self.area1, np.int32)], True, (255,255,0),4)
                self.a1 = len(self.area2)
                cv2.putText(self.frame, str(self.a1),(50,50), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)
                # self.UI.label_result1.setText('{}'.format(self.a1))
                
                self.signal.emit(self.frame)
                self.update_count.emit(str(self.a1))

                if not self.is_running:
                    break

                # Đợi một khoảng thời gian nhỏ
                time.sleep(0.01)

                self.mutex.unlock()

            # Giải phóng tài nguyên và kết thúc luồng
            # self.player.release()


    def stop(self):
        print("stop threading", self.index)
        self.mutex.unlock()
        # self.player.release()
        cv2.destroyAllWindows()
        self.terminate()
    
    def pause_stream(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()

    def resume_stream(self):
        self.mutex.lock()
        self.is_paused = False
        self.wait_cond.wakeAll()
        self.mutex.unlock()
