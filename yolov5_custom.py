from utils.torch_utils import select_device 
from utils.general import check_img_size
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes)
from utils.augmentations import letterbox
import torch 
import cv2 
import numpy as np 
import time 

class Yolov5: 
    def __init__(self):
        self.device = "cpu" # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.weights = "yolov5n.pt" # model path or triton URL
        self.conf = 0.25 # confidence threshold
        self.iou = 0.45 # NMS IOU threshold
        self.imgsz = (640,640) # inference size (height, width)
        self.classes = None # filter by class: --class 0, or --class 0 2 3
        self.names = ""
        self.max_det = 1000  # maximum detections per image
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.half = False  # use FP16 half-precision inference
        self.data = "data/coco128.yaml" # dataset.yaml path
        self.agnostic_nms = False  # class-agnostic NMS
        self.model = None 
        self.agnostic_nms = False
        self.auto = True 
    
    # Load model     
    def set_up_model(self, weights):
        device = select_device(self.device)
        self.weights = weights
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride)
        self.names = self.model.names
        self.device = device
        self.model.eval() # chuyển mô hình sang chế độ evalution để inference
    
    # Convert img to torch 
    # Data Loader 
    def preprocess(self, img):
        img = letterbox(img, self.imgsz, stride=self.model.stride, auto=self.auto)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img
    
    # Hậu xử lý ảnh
    def postprocess(self, preds, img, orig_img):
        preds = non_max_suppression(preds, self.conf, self.iou, agnostic=self.agnostic_nms, max_det=self.max_det, classes=self.classes)
        
        results = []
        for i, det in enumerate(preds):
            if len(det):
                orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
                shape = orig_img.shape
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    results.append([x1, y1, x2, y2, conf, cls])
        return results
        
    # Quá trình inference
    def inference(self, img):
        result = []
        image_copy = img.copy()
        cv2.imwrite("test.jpg", img)
        img = self.preprocess(img)    
        pre_time = time.time()
        preds = self.model(img, augment=False, visualize=False)
        # print("inference_time: ", int((time.time - pre_time)*1000), "ms")
        result = self.postprocess(preds, img, image_copy)
        return result