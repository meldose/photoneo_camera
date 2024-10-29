import numpy as np # imported module np 
import open3d as o3d # imported o3d module
import cv2 # imported cv2 module
import os # imported os module
import sys # imported sys moduel
from sys import platform
from harvesters.core import Harvester
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt") # set the model value 

# Object classes for specific shapes
classNames = ["rectangle", "square", "circle", "oval", "triangle", "polygon"] # set the class names as follows

def display_texture_if_available(texture_component): # defined a function called display texture if it is available or not 
    if texture_component.width == 0 or texture_component.height == 0: # check if the texture component width and height is zero or not
        print("Texture is empty!")
        return

def detect_shapes_with_opencv(color_image): # defined as function for detecting the shapes with opencv with color_image
    # Ensure the image is an 8-bit grayscale image
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Check the data type of the grayscale image
    if gray.dtype != np.uint8: 
        # Normalize to [0, 255] and convert to uint8
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: # checking contour in the list of contours 
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        shape_name = "unidentified"
        if len(approx) == 3: # if the length is 3 then
            shape_name = "triangle" # set shape name as traingle
        elif len(approx) == 4: # if the length is 4 then 
            aspect_ratio = w / float(h) # equation for aspect ration
            shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle" # check if the shape name as square 
        elif len(approx) > 4: # if len is less than 4 then 
            shape_name = "circle" if cv2.isContourConvex(approx) else "oval" # set the shape as circle

        if shape_name in classNames: # if the shape_name is in classNamse
            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2) # draw boundries for the shape created
            cv2.putText(color_image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {shape_name} at {(x, y)}") # print that the object is detected

    cv2.imshow("Detected Shapes", color_image) # show the image 

def display_color_image_with_detection(color_component, name): # defined as function to detect the image with detection
    if color_component.width == 0 or color_component.height == 0:  # check if the color component width and height is zero or not
        print(name + " is empty!")
        return

    # Reshape the image data
    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()

    # Normalize the image to [0, 255] and convert to uint8
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert RGB to BGR
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
                    if detected_class in classNames:  # Only process specified classes
                        print(f"Detected class: {detected_class}, Confidence: {confidence}")
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(color_image, f"{detected_class} {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Call shape detection as well
    detect_shapes_with_opencv(color_image)

    cv2.imshow(name, color_image) # show the image

def software_trigger(): # defined the function called software trigger
    device_id = "TER-008"
    if len(sys.argv) == 2:
        device_id = "PhotoneoTL_DEV_" + sys.argv[1]
    print("--> device_id: ", device_id)

    if platform == "linux":  # if the platform is linux
        cti_file_path_suffix = "/API/lib/photoneo.cti"  # provide the cti file path
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h: # consider the harvester as h 
        h.add_file(cti_file_path, True, True) # add the file
        h.update() # update the harvester

        print("\nName : ID")
        print("---------")
        for item in h.device_info_list: # check if the item is in device list 
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_']) # print the item serial number and its id 

        with h.create({'id_': device_id}) as ia:
            features = ia.remote_device.node_map

            features.PhotoneoTriggerMode.value = "Software"
            features.SendTexture.value = True
            features.SendPointCloud.value = True
            features.SendNormalMap.value = True
            features.SendDepthMap.value = True
            features.SendConfidenceMap.value = True

            ia.start() # start the acquisition

            try: 
                while True: # start an infinite loop 
                    print("\n-- Capturing frame --")
                    features.TriggerFrame.execute()
                    with ia.fetch(timeout=10.0) as buffer:
                        payload = buffer.payload

                        texture_component = payload.components[0]
                        display_texture_if_available(texture_component) # calling the function for displaying the texture 

                        texture_rgb_component = payload.components[1]
                        display_color_image_with_detection(texture_rgb_component, "TextureRGB") # calling the function to display the image with detection

                        point_cloud_component = payload.components[2]
                        norm_component = payload.components[3]

                    if cv2.waitKey(1) & 0xFF == ord('q'): # pressing the key will quit the camera
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop() # stop the acquisition

software_trigger() # calling the function to trigger the software.

