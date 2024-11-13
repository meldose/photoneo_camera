import numpy as np
import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
import torch

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Object classes for specific objects of interest
classNames = ["person", "car"]

def display_texture_if_available(texture_component):
    if texture_component.width == 0 or texture_component.height == 0:
        print("Texture is empty!")
        return
    # Display the texture image
    texture_image = texture_component.data.reshape(texture_component.height, texture_component.width, 3)
    texture_image = cv2.normalize(texture_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Texture Image", texture_image)

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
    
    # Filter detections based on specified classNames
    detections = detections[detections['name'].isin(classNames)]
    
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

    cv2.imshow(name, color_image)

def create_point_cloud_from_component(point_cloud_component):
    # Reshape point cloud data into XYZ coordinates
    points = np.array(point_cloud_component.data).reshape(-1, 3)
    
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def display_and_detect_objects_in_point_cloud(point_cloud):
    # Use DBSCAN clustering for object detection within the point cloud
    labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    max_label = labels.max()

    print(f"Point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # Set outliers to black
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Visualize point cloud with detected clusters
    o3d.visualization.draw_geometries([point_cloud])

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
                        point_cloud = create_point_cloud_from_component(point_cloud_component)
                        display_and_detect_objects_in_point_cloud(point_cloud)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting capture loop.")
                        break
            finally:
                ia.stop()

# Run the software trigger function
if __name__ == "__main__":
    software_trigger()

