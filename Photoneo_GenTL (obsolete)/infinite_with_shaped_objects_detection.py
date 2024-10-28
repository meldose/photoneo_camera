
import numpy as np
import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
from ultralytics import YOLO
import math

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")


# Object classes for specific shapes
classNames = ["rectangle", "square", "circle", "oval"]

# Define display functions

def display_texture_if_available(texture_component): # defining an function for display texture if available
    if texture_component.width == 0 or texture_component.height == 0: # checking if the texture is empty or not
        print("Texture is empty!")
        return
    

def display_pointcloud_if_available(pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp): # defining the function for display pointcloud is available or not
    if pointcloud_comp.width == 0 or pointcloud_comp.height == 0: # check if the texture is empty or not
        print("PointCloud is empty!")
        return
    
    
def display_color_image_with_detection(color_component, name): # defined an function to check if the color image with detection having color component and name
    if color_component.width == 0 or color_component.height == 0: # check if the width and height of the color component is zero
        print(name + " is empty!") # print empty
        return

    # Convert to 3-channel color image
    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy() # convert from 1D to 2D
    color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX) # convert from 2D into 3D 
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

# Apply YOLO object detection
    results = model(color_image, stream=True) # assigning the model having color_image and stream to results.
    for r in results: # checking r in the results
        boxes = r.boxes # setting boxes 
        if boxes: # checking that boxes are there 
            for box in boxes: # checking if box are there in boxes 
                cls = int(box.cls[0]) 
                confidence = box.conf[0]

            # Debugging output
                print(f"Detected class index: {cls}, Confidence: {confidence}")

            # Adjust confidence threshold
                if confidence > 0.5 and cls < len(classNames): # if the box config is greater than 0.5 and length of the class names is less then
                    print(f"Detected class: {classNames[cls]}") # print the Detected class 

                    # x1, y1, x2, y2 = box.xyxy[0]
                    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box and label
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3) # draw the region for rectangle
                    cv2.putText(color_image, f"{classNames[cls]} {confidence:.2f}", (x1, y1 - 10), # put text in color image 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 


    # Display the color image with detections
    cv2.imshow(name, color_image) # display the color image of the object captured.



def software_trigger(): # defined an function for triggering software.
    device_id = "TER-008" # added device id name
    if len(sys.argv) == 2: # check if the len of sys is equal to 2 then
        device_id = "PhotoneoTL_DEV_" + sys.argv[1] 
    print("--> device_id: ", device_id)

    if platform == "linux": # if the platform is linux
        cti_file_path_suffix = "/API/lib/photoneo.cti" # set the cti_file path
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h: # set the harvester as h 
        h.add_file(cti_file_path, True, True) # add the file for harvester
        h.update() # update the harvester

        print("\nName : ID")
        print("---------")
        for item in h.device_info_list: # check if the item is there in the device list or not
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_']) # print the serial number and id of the device 
        print()

        with h.create({'id_': device_id}) as ia:
            features = ia.remote_device.node_map

            # Configure trigger and data streaming
            features.PhotoneoTriggerMode.value = "Software" # se the TriggerMode as software
            features.SendTexture.value = True # set the Texture as True value
            features.SendPointCloud.value = True # set the Pointcloud value as True
            features.SendNormalMap.value = True # set the NormalMap value as True
            features.SendDepthMap.value = True # set the DepthMap value as True
            features.SendConfidenceMap.value = True # set the confidenceMap as True

            ia.start() # start the acquisition 

            while True: # set an infinite loop
                print("\n-- Capturing frame --")
                features.TriggerFrame.execute() # se the Trigger frame 
                with ia.fetch(timeout=10.0) as buffer:
                    payload = buffer.payload

                    # Display texture
                    texture_component = payload.components[0]
                    display_texture_if_available(texture_component)

                    # Display color image with YOLO detection
                    texture_rgb_component = payload.components[1]
                    display_color_image_with_detection(texture_rgb_component, "TextureRGB")

                    # Display 3D point cloud if required
                    point_cloud_component = payload.components[2]
                    norm_component = payload.components[3]
                    display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)

                # Press 'q' to quit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting capture loop.")
                    break

            ia.stop() # stop the camera

software_trigger() # calling the function for software trigger.

