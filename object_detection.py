

from ultralytics import YOLO # imported YOLO from ultralytics
import cv2 # imported cv2 module
import math #  imported the math module
from ultralytics import YOLO
model = YOLO("yolo-Weights/yolov8n.pt") # assigning the model variable as YOLO path having the image.

cap = cv2.VideoCapture(0) # starting the Webcamera
cap.set(3, 640)
cap.set(4, 480)


# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True: # checking the condition
    success, img = cap.read() # if success then read the captured image
    results = model(img, stream=True) # assigning the results as model having image and stream

    # coordinates
    for r in results: # checking the condition for r in results
        boxes = r.boxes # assigning the boxes as r.boxes

        for box in boxes: # checking the condition for box in boxes
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0] # assigning the x1, y1, x2, y2 to the box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # draw the rectangle

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100 # assign the confidence as math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence) # print the confidence

            # class name
            cls = int(box.cls[0]) # assign the cls as int(box.cls[0])
            print("Class name -->", classNames[cls]) # print the class name

            # object details
            org = [x1, y1] # assigning the org as x1, y1
            font = cv2.FONT_HERSHEY_SIMPLEX # assigning the font as cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1 # assigning the fontScale as 1
            color = (255, 0, 0) # assigning the color as (255, 0, 0)
            thickness = 2 # assigning the thickness as 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness) # put text in the image

    cv2.imshow('Webcam', img) # show the webcam image
    if cv2.waitKey(1) == ord('q'): # by pressing the Wait-key as q it will close
        break

cap.release() # release the camera
cv2.destroyAllWindows() # destroy all the windows created while capturing 