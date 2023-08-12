
import numpy as np
import classifier

import cv2



car_make_classifier = classifier.Classifier('model-weights-spectrico-car-brand-recognition-mobilenet_v3-224x224-170620.mnn', 'labels-makes.txt')

image_path = 'test.jpg'

# Đọc hình ảnh
img = cv2.imread(image_path)



if img is not None:
    make = car_make_classifier.predict(img)
    # Tiến hành dự đoán trên hình ảnh
    # (make, make_confidence) = car_make_classifier.predict(img)

    
    # In kết quả dự đoán
    print("Thương hiệu xe: ", make)

else:
    print("Không thể đọc hình ảnh")