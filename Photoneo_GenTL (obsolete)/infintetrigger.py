import numpy as np  # imported numpy module
import open3d as o3d  # imported open 3d module
import cv2  # imported cv2 module
import os  # imported os 
import sys  # imported sys 
from sys import platform  # imported platform module from sys class
from harvesters.core import Harvester

def display_texture_if_available(texture_component): # defining an function for display texture if available
    if texture_component.width == 0 or texture_component.height == 0: # checking if the texture is empty or not
        print("Texture is empty!")
        return

    texture = texture_component.data.reshape(texture_component.height, texture_component.width, 1).copy() # copying the texture component height and height
    texture_screen = cv2.normalize(texture, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX) #  creating an 2D image
    cv2.imshow("Texture", texture_screen)# displaying the 2D image texture.
    return

def display_color_image_if_available(color_component, name): # defining the function for color image available or not
    if color_component.width == 0 or color_component.height == 0: # checking if the texture is empty or not
        print(name + " is empty!")
        return

    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy() # creating an 3D image
    color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX) # opening the 3D image in an camera
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, color_image) # show the 3D image
    return

def display_pointcloud_if_available(pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp): # defining the function for display pointcloud is available or not
    if pointcloud_comp.width == 0 or pointcloud_comp.height == 0: # check if the texture is empty or not
        print("PointCloud is empty!")
        return

    pointcloud = pointcloud_comp.data.reshape(pointcloud_comp.height * pointcloud_comp.width, 3).copy() # copying the pointcloud to reshape the height and width
    pcd = o3d.geometry.PointCloud() # assigning the pointcloud to the variable pcd 
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    if normal_comp.width > 0 and normal_comp.height > 0: # if the normal width and height are greater than zero
        norm_map = normal_comp.data.reshape(normal_comp.height * normal_comp.width, 3).copy()
        pcd.normals = o3d.utility.Vector3dVector(norm_map) # copying the 3d shapes 

    texture_rgb = np.zeros((pointcloud_comp.height * pointcloud_comp.width, 3))
    if texture_comp.width > 0 and texture_comp.height > 0: # if the texture component widht and height is greater than zero then 
        texture = texture_comp.data.reshape(texture_comp.height, texture_comp.width, 1).copy() # copy the component height and width 
        texture_rgb[:, 0] = np.reshape(1 / 65536 * texture, -1)
        texture_rgb[:, 1] = np.reshape(1 / 65536 * texture, -1)
        texture_rgb[:, 2] = np.reshape(1 / 65536 * texture, -1)
    elif texture_rgb_comp.width > 0 and texture_rgb_comp.height > 0:
        texture = texture_rgb_comp.data.reshape(texture_rgb_comp.height, texture_rgb_comp.width, 3).copy() # if the texture rgb comp width and height is greater than zero then
        texture_rgb[:, 0] = np.reshape(1 / 65536 * texture[:, :, 0], -1)
        texture_rgb[:, 1] = np.reshape(1 / 65536 * texture[:, :, 1], -1)
        texture_rgb[:, 2] = np.reshape(1 / 65536 * texture[:, :, 2], -1)
    else:
        print("Texture and TextureRGB are empty!")
        return
    texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # creating 3D image
    pcd.colors = o3d.utility.Vector3dVector(texture_rgb)
    o3d.visualization.draw_geometries([pcd], width=800, height=600) # show 3D images 
    return

def software_trigger():  # Continuous mode, removed 'iterations' parameter
    device_id = "TER-008" # device_id created
    if len(sys.argv) == 2:# if the length of sys is 2 then
        device_id = "PhotoneoTL_DEV_" + sys.argv[1]
    print("--> device_id: ", device_id)

    if platform == "linux": # if the platform is linux then
        cti_file_path_suffix = "/API/lib/photoneo.cti"#  assign the cti file path suffix as follows.
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h:
        h.add_file(cti_file_path, True, True) # adding the file path to correct location
        h.update()

        print()
        print("Name : ID")
        print("---------")
        for item in h.device_info_list: # checking if the item is in the info list or not
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_']) # printing the serial_number with item property dict
        print()

        with h.create({'id_': device_id}) as ia:# creating the device_id with harvester
            features = ia.remote_device.node_map

            print("TriggerMode BEFORE: ", features.PhotoneoTriggerMode.value) 
            features.PhotoneoTriggerMode.value = "Software" # assigning the Photoneo_TriggerMode as Software
            print("TriggerMode AFTER: ", features.PhotoneoTriggerMode.value)

            features.SendTexture.value = True
            features.SendPointCloud.value = True
            features.SendNormalMap.value = True
            features.SendDepthMap.value = True
            features.SendConfidenceMap.value = True

            ia.start()

            while True:  # Run indefinitely
                print("\n-- Capturing frame --")
                features.TriggerFrame.execute()  # trigger frame
                with ia.fetch(timeout=10.0) as buffer:
                    payload = buffer.payload

                    texture_component = payload.components[0]
                    display_texture_if_available(texture_component)

                    texture_rgb_component = payload.components[1]
                    display_color_image_if_available(texture_rgb_component, "TextureRGB")
                    color_image_component = payload.components[7]
                    display_color_image_if_available(color_image_component, "ColorCameraImage")

                    point_cloud_component = payload.components[2]
                    norm_component = payload.components[3]
                    display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)

                # Add an exit option to stop the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
                    print("Exiting capture loop.")
                    break

            ia.stop()  # Stop the acquisition after exiting the loop

software_trigger() # calling the function