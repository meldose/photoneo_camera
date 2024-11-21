
import numpy as np # imported numpy module as np    
import cv2 # imported cv2 module
import os # imported os module
from sys import platform # imported sys module 
from harvesters.core import Harvester

# Object classes for specific shapes and other objects of interest
classNames = ["rectangle", "square", "circle", "oval", "triangle", "polygon"]

def display_texture_if_available(texture_component):  # Display texture if available
    if texture_component.width == 0 or texture_component.height == 0: # checking if the component width and height is zero
        print("Texture is empty!") # print as empty
        return 

def detect_shapes_with_opencv(color_image):  # Detect shapes with OpenCV
    # Ensure the image is an 8-bit grayscale image
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: # checking if the contour is in contours
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        
        shape_name = "unidentified" 
        if len(approx) == 3: # if len is 3 then
            shape_name = "triangle"
        elif len(approx) == 4: # if len is 4 then
            aspect_ratio = w / float(h)
            shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle" #  if aspec ratio is less that 0.9 and less than 1.1 or else it is rectangle
        elif len(approx) > 4: # if the len is greater than 4 
            shape_name = "circle" if cv2.isContourConvex(approx) else "oval" # set the figure as circle or as oval

        if shape_name in classNames:
            # Draw contour and display shape name and dimensions
            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
            dimension_label = f"{shape_name} W:{w}px H:{h}px"
            cv2.putText(color_image, dimension_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {shape_name} at {(x, y)} with dimensions {w}px x {h}px")

def display_color_image_with_detection(color_component, name):  # Display and detect shapes
    if color_component.width == 0 or color_component.height == 0: # if the component widht /height is zero then
        print(name + " is empty!")
        return

    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy() # reshaping the items into 3D format 
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # Detect shapes within the image using OpenCV
    detect_shapes_with_opencv(color_image)

    cv2.imshow(name, color_image) # show the image

def software_trigger(): # function for software trigger
    ####################
    device_id = "PhotoneoTL_DEV_TER-008"
    print("--> device_id: ", device_id)

    if platform == "linux":
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h: # adding the harvester 
        h.add_file(cti_file_path, True, True) # adding the file 
        h.update() # updating the harvester file

        print("\nName : ID")
        print("---------")
        for item in h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])

        with h.create({'id_': device_id}) as ia:
            features = ia.remote_device.node_map # determing all the features

            features.PhotoneoTriggerMode.value = "Software"
            features.SendTexture.value = True
            features.SendPointCloud.value = True
            features.SendNormalMap.value = True
            features.SendDepthMap.value = True
            features.SendConfidenceMap.value = True

            ia.start() # start the harvester 

            try:
                while True:
                    print("\n-- Capturing frame --")
                    features.TriggerFrame.execute() # triggering the frame work 
                    with ia.fetch(timeout=10.0) as buffer:
                        payload = buffer.payload

                        texture_component = payload.components[0]
                        display_texture_if_available(texture_component)

                        texture_rgb_component = payload.components[1]
                        display_color_image_with_detection(texture_rgb_component, "TextureRGB") # calling the function for display_color_image_with_detection having Texture RGB

                        point_cloud_component = payload.components[2]
                        norm_component = payload.components[3]

                    if cv2.waitKey(1) & 0xFF == ord('q'): # close the texture if q pressed
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop() # stop the aquisition

# Run the software trigger function
if __name__ == "__main__":
    software_trigger() # calling the software trigger.


