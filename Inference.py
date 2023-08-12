import cv2
import numpy as np
import time
from  queue import Queue
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from tracker import *
import torch
import sys, os
from PIL import Image
import torchvision.transforms as transforms

# from yolov5.models.common import *
from detect_yolov5 import Detection, Tracking
from predict_yolo import Predict




class Inference(QThread):
    signal = pyqtSignal(np.ndarray)
    update_count = pyqtSignal(str)
    reset_signal = pyqtSignal(bool)

    CAR_signal = pyqtSignal(np.ndarray, str, str)
    # area1 = [(1120,1000),(1128,1162),(1336,1180),(1284,1019)]
    area2 = set()
    count = 0
    tracker = Tracker()
    tracking = Tracking()
    
    def __init__(self, area1):
        super().__init__()
        self.list_car_img=[]
        self.frame = None
        self.threadActive = False

        self.device = None
        self.out_file = None
        self.classes = None
        self.model = None

        self.area1 = area1

        self.mutex = QMutex()
        self.is_paused = False
        self.is_running = True
        self.wait_cond = QWaitCondition()
        self.detection= Detection()
        self.detection.weights = 'yolov5s.pt'

        self.transform = None
        self.transform1 = None
        self.clf = None
        self.clf_brand = Predict()
        self.clf_brand.weights = 'best1.pt'
        self.color_brand = Predict()
        self.color_brand.weights = 'color.pt'

        self.dict_color = {0:'black',1: 'blue', 2:'cyan', 3:'gray', 4:'green', 5:'red', 6:'white', 7:'yellow'}
        
        self.queue = Queue()
        


    # def load_color_model(self):
    #     self.clf = torch.load('color_model2.pt', map_location=torch.device('cpu'))
    #     # self.clf = self.clf.cuda()
    #     self.clf.eval()
        

    # def transform_img(self):
    #     size = 150
    #     self.transform = transforms.Compose([
    #         transforms.Resize((size, size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.4344, 0.4025, 0.3941], std=[0.1718, 0.1622, 0.1627])
    #     ])
        
    
    # def frame_to_img(self,frame):
    #     image = Image.fromarray(frame)
    #     input_image = self.transform(image).unsqueeze(0) 
    #     # input_image = input_image.cuda()
    #     return input_image

    # def color_recog(self, input_image):
    #     with torch.no_grad():
    #         output = self.clf(input_image)
    #     probabilities = torch.softmax(output, dim=1)
    #     predicted_label = torch.argmax(probabilities, dim=1).item()
    #     color  = self.dict_color.get(predicted_label)
        
    #     return color

    def run(self):
        
        print('Bắt đầu Inference!')
        self.area2 = set()
        self.threadActive = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.out_file = 'labeled_video.avi'
        self.detection._load_model()
        self.clf_brand._load_model()
        self.color_brand._load_model()
        # self.load_color_model()
        # self.transform_img()

        while self.threadActive:
            self.frame = self.queue.get()
            if self.frame is not None:

                self.mutex.lock()
                while self.is_paused:
                    self.wait_cond.wait(self.mutex)

               
                
                lst = []
                result11 = self.detection.detect(self.frame)
                for box in result11:
                    x1 ,y1, x2, y2 , cls, conf = box 


                    lst.append([x1,y1,x2,y2])
                cv2.rectangle(self.frame,(x1,y1),(x2,y2), (0,0,255),2)
                idx_bbox = self.tracker.update(lst)
                for bbox in idx_bbox:
                    x3,y3,x4,y4,id = bbox
                    cv2.rectangle(self.frame,  (x3,y3),(x4,y4), (0,0,255),2)
                    cv2.putText(self.frame, str(id),(x3,y3), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
                    central_box = ((x3+x4)//2,(y3+y4)//2)
                    cv2.circle(self.frame, (central_box),4,(0,255,0),-1)
                    
                    result1 = cv2.pointPolygonTest(np.array(self.area1,np.int32), (((x3+x4)//2,(y3+y4)//2)),False)        
                    if result1 >0:
                        
                        self.car_img = self.frame[y3:y4,x3:x4]
                        self.car_color= self.color_brand.predict(self.car_img)
                        self.car_brand = self.clf_brand.predict(self.car_img)
                        # self.car_brand = 'Noneeeeee'

                        cv2.putText(self.frame, str(self.car_color),(x4,y4), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
                        
                        self.CAR_signal.emit(self.car_img, self.car_color, self.car_brand)
                        self.area2.add(id)
                        

                # cv2.polylines(self.frame,[np.array(self.area1, np.int32)], True, (255,255,0),4)
                self.a1 = len(self.area2)
                cv2.putText(self.frame, str(self.a1),(50,50), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)

                
                self.signal.emit(self.frame)
                self.update_count.emit(str(self.a1))

                if not self.is_running:
                    break

                # Đợi một khoảng thời gian nhỏ
                time.sleep(0.001)

                self.mutex.unlock()

            # Giải phóng tài nguyên và kết thúc luồng
            # self.player.release()


    def stop(self):
        
        self.mutex.lock()
        self.threadActive = False
        self.mutex.unlock()
    
    def pause_stream(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()

    def resume_stream(self):
        self.mutex.lock()
        self.is_paused = False
        self.wait_cond.wakeAll()
        self.mutex.unlock()
