import numpy as np # imported numpy as np 
import open3d as o3d # imported open3d module as o3d
import cv2 # imported module cv2
import os # imported os module
import sys # imported sys module
from sys import platform
from harvesters.core import Harvester
import torch 

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Object classes for specific shapes and other objects of interest
classNames = ["rectangle", "square", "circle", "oval", "triangle", "polygon", "person", "car"]

def display_texture_if_available(texture_component): # defining an function for displaying texture 
    if texture_component.width == 0 or texture_component.height == 0: # if texture component width and height is zero 
        print("Texture is empty!") # print texture is empty
        return 

def detect_shapes_with_opencv(color_image): # defining an function for detecting shapes with opencv
    # Ensure the image is an 8-bit grayscale image
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) 
    if gray.dtype != np.uint8: 
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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
            # Draw contour and display shape name and dimensions
            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
            dimension_label = f"{shape_name} W:{w}px H:{h}px"
            cv2.putText(color_image, dimension_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {shape_name} at {(x, y)} with dimensions {w}px x {h}px")

def display_color_image_with_detection(color_component, name):
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return

    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection on the image
    results = model(color_image)
    detections = results.pandas().xyxy[0]
    
    # Display bounding boxes and dimensions for YOLO-detected objects
    for idx, row in detections.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        width = x_max - x_min
        height = y_max - y_min
        label = f"{row['name']} {row['confidence']:.2f}"
        dimensions_label = f"W:{width}px H:{height}px"

        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(color_image, dimensions_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected {label} at {(x_min, y_min)} with dimensions {width}px x {height}px")

    # Detect shapes within the image using OpenCV
    detect_shapes_with_opencv(color_image)

    cv2.imshow(name, color_image)

def software_trigger():
    device_id = "TER-008"
    if len(sys.argv) == 2:
        device_id = "PhotoneoTL_DEV_" + sys.argv[1]
    print("--> device_id: ", device_id)

    if platform == "linux":
        cti_file_path_suffix = "/API/lib/photoneo.cti"
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

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop()

# Run the software trigger function
if __name__ == "__main__":
    software_trigger()

