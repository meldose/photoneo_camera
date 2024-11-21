
import numpy as np # imported numpy as np   
import cv2 # imported cv2 module
import os # imported os module
from sys import platform # imported sys module
from harvesters.core import Harvester

# Object classes for specific shapes and other objects of interest
classNames = ["rectangle", "square", "circle", "oval", "triangle", "polygon"]
################################################################################################################
def display_texture_if_available(texture_component):  # Display texture if available
    if texture_component.width == 0 or texture_component.height == 0: 
        print("Texture is empty!")
        return 
########################################################################################################################
def detect_shapes_with_opencv(color_image):  # Detect shapes with OpenCV
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
            cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
            dimension_label = f"{shape_name} W:{w}px H:{h}px"
            cv2.putText(color_image, dimension_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f"Detected {shape_name} at {(x, y)} with dimensions {w}px x {h}px")
######################################################################################################################################

def display_color_image_with_detection(color_component, name):  # Display and detect shapes
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return

    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    detect_shapes_with_opencv(color_image)

    cv2.imshow(name, color_image)

######################################################################################################################################
def save_point_cloud(point_cloud_component, file_name="point_cloud.ply"):

    if point_cloud_component.width == 0 or point_cloud_component.height == 0: # if the width and height of the point_cloud compoent is zero
        print("Point cloud is empty!")
        return
    
    point_cloud_data = point_cloud_component.data.reshape(point_cloud_component.height, point_cloud_component.width, -1)
    print(f"Point cloud shape: {point_cloud_data.shape}")
    
    xyz = point_cloud_data[:, :, :3]  # Assuming format is [X, Y, Z, ...]
    xyz = xyz.reshape(-1, 3)

    with open(file_name, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {xyz.shape[0]}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        for x, y, z in xyz:
            ply_file.write(f"{x} {y} {z}\n")
    print(f"Point cloud saved to {file_name}")

#########################################################################################################################################

def software_trigger_with_pointcloud():
    device_id = "PhotoneoTL_DEV_TER-008"
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
                        save_point_cloud(point_cloud_component, "output_point_cloud.ply")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop()

if __name__ == "__main__":
    software_trigger_with_pointcloud()


############################################# calling script #########################################################################################################
# import numpy as np
# import cv2
# import os
# from sys import platform
# from harvesters.core import Harvester

# # Object classes for specific shapes and other objects of interest
# classNames = ["rectangle", "square", "circle", "oval", "triangle", "polygon"]

# def display_texture_if_available(texture_component):
#     """ Display texture if available. """
#     if texture_component.width == 0 or texture_component.height == 0:
#         print("Texture is empty!")
#         return

# def detect_shapes_with_opencv(color_image):
#     """ Detect shapes in the image using OpenCV. """
#     gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#     if gray.dtype != np.uint8:
#         gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edged = cv2.Canny(blurred, 50, 150)
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     shape_data = []
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
#         x, y, w, h = cv2.boundingRect(approx)
        
#         shape_name = "unidentified"
#         if len(approx) == 3:
#             shape_name = "triangle"
#         elif len(approx) == 4:
#             aspect_ratio = w / float(h)
#             shape_name = "square" if 0.9 <= aspect_ratio <= 1.1 else "rectangle"
#         elif len(approx) > 4:
#             shape_name = "circle" if cv2.isContourConvex(approx) else "oval"

#         if shape_name in classNames:
#             shape_data.append({
#                 "shape": shape_name,
#                 "position": (x, y),
#                 "dimensions": (w, h)
#             })
#             print(f"Detected {shape_name} at {(x, y)} with dimensions {w}px x {h}px")
#     return shape_data

# def save_point_cloud(point_cloud_component, file_name="point_cloud.ply"):
#     """ Save the point cloud data to a PLY file. """
#     if point_cloud_component.width == 0 or point_cloud_component.height == 0:
#         print("Point cloud is empty!")
#         return
    
#     point_cloud_data = point_cloud_component.data.reshape(point_cloud_component.height, point_cloud_component.width, -1)
#     print(f"Point cloud shape: {point_cloud_data.shape}")
    
#     xyz = point_cloud_data[:, :, :3]  # Assuming format is [X, Y, Z, ...]
#     xyz = xyz.reshape(-1, 3)

#     with open(file_name, 'w') as ply_file:
#         ply_file.write("ply\n")
#         ply_file.write("format ascii 1.0\n")
#         ply_file.write(f"element vertex {xyz.shape[0]}\n")
#         ply_file.write("property float x\n")
#         ply_file.write("property float y\n")
#         ply_file.write("property float z\n")
#         ply_file.write("end_header\n")
#         for x, y, z in xyz:
#             ply_file.write(f"{x} {y} {z}\n")
#     print(f"Point cloud saved to {file_name}")

# def trigger_process(image_component, point_cloud_component, image_callback, point_cloud_callback):
#     """ Trigger the processing of image and point cloud data. """
#     # Trigger image processing
#     if image_component and image_component.width > 0 and image_component.height > 0:
#         color_image = image_component.data.reshape(image_component.height, image_component.width, 3).copy()
#         color_image = cv2.normalize(color_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
#         image_callback(color_image)

#     # Trigger point cloud saving
#     if point_cloud_component and point_cloud_component.width > 0 and point_cloud_component.height > 0:
#         save_point_cloud(point_cloud_component)

# def software_trigger_with_pointcloud():
#     """ Main function to trigger software capture and process data. """
#     device_id = "PhotoneoTL_DEV_TER-008"
#     print("--> device_id: ", device_id)

#     if platform == "linux":
#         cti_file_path_suffix = "/API/lib/photoneo.cti"
#     else:
#         cti_file_path_suffix = "/API/lib/photoneo.cti"
#     cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
#     print("--> cti_file_path: ", cti_file_path)

#     with Harvester() as h:
#         h.add_file(cti_file_path, True, True)
#         h.update()

#         print("\nName : ID")
#         print("---------")
#         for item in h.device_info_list:
#             print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])

#         with h.create({'id_': device_id}) as ia:
#             features = ia.remote_device.node_map

#             features.PhotoneoTriggerMode.value = "Software"
#             features.SendTexture.value = True
#             features.SendPointCloud.value = True
#             features.SendNormalMap.value = True
#             features.SendDepthMap.value = True
#             features.SendConfidenceMap.value = True

#             ia.start()

#             try:
#                 while True:
#                     print("\n-- Capturing frame --")
#                     features.TriggerFrame.execute()
#                     with ia.fetch(timeout=10.0) as buffer:
#                         payload = buffer.payload

#                         texture_component = payload.components[0]
#                         texture_rgb_component = payload.components[1]
#                         point_cloud_component = payload.components[2]

#                         # Trigger the processing using callbacks
#                         trigger_process(texture_rgb_component, point_cloud_component, detect_shapes_with_opencv, save_point_cloud)

#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         print("Exiting capture loop.")
#                         break
#             finally:
#                 ia.stop()

# if __name__ == "__main__":
#     software_trigger_with_pointcloud()
