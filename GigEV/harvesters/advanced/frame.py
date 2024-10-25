
################CAMERA FPS TEST#############################

import cv2
import time

def measure_fps(camera_index=0, num_frames=100):
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)

    # Verify if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Start time
    start_time = time.time()

    # Capture frames
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

    # End time
    end_time = time.time()

    # Release the camera
    cap.release()

    # Calculate FPS
    elapsed_time = end_time - start_time
    fps = num_frames / elapsed_time
    print(f"Approximate FPS: {fps:.2f}")
    return fps

# Call the function to measure FPS
measure_fps()

##############TEST WITH HARDWARE TRIGGER########################

import sys
import time
import numpy as np
import open3d as o3d
import cv2
import os
from pathlib import Path
from sys import platform
from typing import List
from genicam.genapi import NodeMap
from harvesters.core import Component2DImage, Harvester
from photoneo_genicam.default_gentl_producer import producer_path
from photoneo_genicam.features import enable_hardware_trigger
from photoneo_genicam.user_set import load_default_user_set

def display_texture_if_available(texture_component):
    if texture_component.width == 0 or texture_component.height == 0:
        print("Texture is empty!")
        return
    
    texture = texture_component.data.reshape(texture_component.height, texture_component.width, 1).copy()
    texture_screen = cv2.normalize(texture, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    cv2.imshow("Texture", texture_screen)
    return

def display_color_image_if_available(color_component, name):
    if color_component.width == 0 or color_component.height == 0:
        print(name + " is empty!")
        return
    
    color_image = color_component.data.reshape(color_component.height, color_component.width, 3).copy()
    color_image = cv2.normalize(color_image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, color_image)
    return

def display_pointcloud_if_available(pointcloud_comp, normal_comp, texture_comp, texture_rgb_comp):
    if pointcloud_comp.width == 0 or pointcloud_comp.height == 0:
        print("PointCloud is empty!")
        return
    
    pointcloud = pointcloud_comp.data.reshape(pointcloud_comp.height * pointcloud_comp.width, 3).copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    if normal_comp.width > 0 and normal_comp.height > 0:
        norm_map = normal_comp.data.reshape(normal_comp.height * normal_comp.width, 3).copy()
        pcd.normals = o3d.utility.Vector3dVector(norm_map)

    texture_rgb = np.zeros((pointcloud_comp.height * pointcloud_comp.width, 3))
    if texture_comp.width > 0 and texture_comp.height > 0:
        texture = texture_comp.data.reshape(texture_comp.height, texture_comp.width, 1).copy()
        texture_rgb[:, 0] = np.reshape(1 / 65536 * texture, -1)
        texture_rgb[:, 1] = np.reshape(1 / 65536 * texture, -1)
        texture_rgb[:, 2] = np.reshape(1 / 65536 * texture, -1)        
    elif texture_rgb_comp.width > 0 and texture_rgb_comp.height > 0:
        texture = texture_rgb_comp.data.reshape(texture_rgb_comp.height, texture_rgb_comp.width, 3).copy()
        texture_rgb[:, 0] = np.reshape(1 / 65536 * texture[:, :, 0], -1)
        texture_rgb[:, 1] = np.reshape(1 / 65536 * texture[:, :, 1], -1)
        texture_rgb[:, 2] = np.reshape(1 / 65536 * texture[:, :, 2], -1)
    else:
        print("Texture and TextureRGB are empty!")
        return

    texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    pcd.colors = o3d.utility.Vector3dVector(texture_rgb)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)
    return

def measure_fps_and_capture(device_sn: str, num_frames: int = 100):
    try:
        with Harvester() as h:
            h.add_file(str(producer_path), check_existence=True, check_validity=True)
            h.update()

            print(f"Connecting to device with serial number: {device_sn}")
            with h.create({"serial_number": device_sn}) as ia:
                features: NodeMap = ia.remote_device.node_map

                load_default_user_set(features)
                enable_hardware_trigger(features)

                ia.start()
                timeout = 180
                print(f"Waiting {timeout}s for hardware-trigger signal...")

                start_time = time.time()

                for i in range(num_frames):
                    features.TriggerFrame.execute()  # Trigger frame
                    with ia.fetch(timeout=timeout) as buffer:
                        payload = buffer.payload

                        # Process components
                        texture_component = payload.components[0]
                        display_texture_if_available(texture_component)
                        
                        texture_rgb_component = payload.components[1]
                        display_color_image_if_available(texture_rgb_component, "TextureRGB")

                        point_cloud_component = payload.components[2]
                        norm_component = payload.components[3]
                        display_pointcloud_if_available(point_cloud_component, norm_component, texture_component, texture_rgb_component)

                end_time = time.time()

                elapsed_time = end_time - start_time
                fps = num_frames / elapsed_time
                print(f"Approximate FPS: {fps:.2f}")
                return fps

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        device_id = sys.argv[1]
        num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # Default to 100 frames
        print(f"Device ID: {device_id}, Number of frames: {num_frames}")
    except IndexError:
        print("Error: No device given, please run with the device serial number as argument:")
        print(f"    {Path(__file__).name} <device serial> [num_frames]")
        sys.exit(1)
    except ValueError:
        print("Error: Number of frames must be an integer.")
        sys.exit(1)

    if not device_id:
        print("Error: Device ID cannot be None.")
        sys.exit(1)

    measure_fps_and_capture(device_id, num_frames)


