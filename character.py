import os
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from sort.sort import Sort
# from ..config import ROOT
import time


class Detection_char:
    def __init__(self):
        # self.weights = os.path.join(ROOT, "/resources/Weight/face_v3.pt")
        # self.weights = os.path.join("character_model.pt")
        self.imgsz = (416, 416)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cuda'
        self.classes = None
        self.agnostic_nms = True
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def _load_model(self):
        # Load model
        # self.device = select_device(self.device)
        if self.device == "cpu":
            arg = "cpu"
        else:
            arg = f"{self.device}"
        self.device = torch.device(arg)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, fp16=self.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def detect(self, image):
        bboxes = []
        class_name = []
        im = letterbox(image, self.imgsz, stride=self.stride,
                       auto=self.pt)[0]  # resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):
            if len(det):
                det = det.detach().cpu().numpy()
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    class_names = self.names[int(cls)]
                    class_name.append(class_names)
                    bboxes.append([x1, y1, x2, y2, class_names, float(conf)])
        
        threshold = 12  # Ngưỡng khoảng cách tọa độ y
        groups = []  # Danh sách các nhóm bounding box trên các hàng

        # Phân nhóm các bounding box trên các hàng
        for bbox in bboxes:
            matched_group = False
            for group in groups:
                if abs(bbox[1] - group[0][1]) <= threshold:
                    group.append(bbox)
                    matched_group = True
                    break
            if not matched_group:
                groups.append([bbox])

        # Sắp xếp từng nhóm theo tọa độ x
        for group in groups:
            group.sort(key=lambda x: x[0])

        # Trích xuất tên lớp từ các bounding box đã sắp xếp
        class_name_sorted = []

        # Kiểm tra số nhóm và xử lý tương ứng
        if len(groups) == 1:
            # Trường hợp chỉ có 1 dòng
            class_name_sorted = [x[4] for x in groups[0]]
        else:
            # Trường hợp có 2 dòng
            for i, group in enumerate(groups):
                class_name_sorted.extend([x[4] for x in group])
                if i != len(groups) - 1:
                    class_name_sorted.append("-")
        return class_name_sorted