import numpy as np
import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
from ultralytics import YOLO
import math

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes for specific shapes
classNames = ["rectangle", "square", "circle", "oval"]

def display_texture_if_available(texture_component):
    if texture_component.width == 0 or texture_component.height == 0:
        print("Texture is empty!")
        return

def detect_shapes_with_opencv(color_image):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        shape_name = "unidentified"
        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) == 4:
            aspect_ratio = w / float(h)
            shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
        elif len(approx) > 4:
            shape_name = "circle" if cv2.isContourConvex(approx) else "oval"

        if shape_name in classNames:
            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(color_image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {shape_name} at {(x, y)}")

    cv2.imshow("Detected Shapes", color_image)

def display_pointcloud_if_available(pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp):
    if pointcloud_comp.width == 0 or pointcloud_comp.height == 0:
        print("PointCloud is empty!")
        return

def display_color_image_with_detection(color_component, name):
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return

    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    results = model(color_image, stream=True)
    for r in results:
        boxes = r.boxes
        if boxes:
            for box in boxes:
                cls = int(box.cls[0])
                confidence = box.conf[0]

                if confidence > 0 and cls < len(classNames):
                    detected_class = classNames[cls]
                    if detected_class not in classNames:
                        continue

                    print(f"Detected class: {detected_class}, Confidence: {confidence}")
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(color_image, f"{detected_class} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(name, color_image)

def software_trigger():
    device_id = "TER-008"
    if len(sys.argv) == 2:
        device_id = "PhotoneoTL_DEV_" + sys.argv[1]
    print("--> device_id: ", device_id)

    if platform == "linux": # if the platform is linux
        cti_file_path_suffix = "/API/lib/photoneo.cti" # provide the cti file path
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h:
        h.add_file(cti_file_path, True, True)
        h.update()

        print("\nName : ID")
        print("---------")
        for item in h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])

        with h.create({'id_': device_id}) as ia:
            features = ia.remote_device.node_map

            features.PhotoneoTriggerMode.value = "Software"
            features.SendTexture.value = True
            features.SendPointCloud.value = True
            features.SendNormalMap.value = True
            features.SendDepthMap.value = True
            features.SendConfidenceMap.value = True

            ia.start()

            try:
                while True:
                    print("\n-- Capturing frame --")
                    features.TriggerFrame.execute()
                    with ia.fetch(timeout=10.0) as buffer:
                        payload = buffer.payload

                        texture_component = payload.components[0]
                        display_texture_if_available(texture_component)

                        texture_rgb_component = payload.components[1]
                        display_color_image_with_detection(texture_rgb_component, "TextureRGB")

                        point_cloud_component = payload.components[2]
                        norm_component = payload.components[3]
                        display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop()

software_trigger()
