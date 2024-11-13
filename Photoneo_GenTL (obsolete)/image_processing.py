import numpy as np
import open3d as o3d
import cv2
import os
import sys
from harvesters.core import Harvester
import torch
import matplotlib.pyplot as plt

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Object classes for specific objects of interest
classNames = ["person", "car"]

def display_texture_if_available(texture_component):
    if texture_component.width == 0 or texture_component.height == 0:
        print("Texture component is empty or unavailable!")
        return
    try:
        # Reshape the data based on component's width, height, and channels (assuming 3 channels for RGB)
        texture_image = texture_component.data.reshape(texture_component.height, texture_component.width, 3)
        
        # Normalize and convert to 8-bit if necessary
        texture_image = cv2.normalize(texture_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert color format if needed (OpenCV uses BGR by default)
        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR)
        
        # Display or save the texture image
        if os.getenv("DISPLAY") is None:
            cv2.imwrite("TextureImage.png", texture_image)
            print("Texture image saved as TextureImage.png")
        else:
            cv2.imshow("Texture Image", texture_image)
            cv2.waitKey(1)  # Ensures the OpenCV window refreshes
    except Exception as e:
        print(f"Error displaying texture image: {e}")

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
    detections = detections[detections['name'].isin(classNames)]

    # Display bounding boxes
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

    if os.getenv("DISPLAY") is None:
        cv2.imwrite(f"{name}.png", color_image)
    else:
        cv2.imshow(name, color_image)

def create_point_cloud_from_component(point_cloud_component):
    if point_cloud_component.data.size == 0:
        print("Point cloud component is empty!")
        return None
    points = np.array(point_cloud_component.data).reshape(-1, 3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def display_and_detect_objects_in_point_cloud(point_cloud):
    if point_cloud is None or len(point_cloud.points) == 0:
        print("Point cloud is empty or has no points.")
        return
    labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    if labels.size == 0:
        print("No clusters detected in point cloud.")
        return
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([point_cloud])

def software_trigger():
    device_id = "TER-008"
    if len(sys.argv) == 2:
        device_id = "PhotoneoTL_DEV_" + sys.argv[1]
    print("--> device_id: ", device_id)
    if os.name == "posix":
        cti_file_path_suffix = "/API/lib/photoneo.cti"
    else:
        cti_file_path_suffix = "\\API\\lib\\photoneo.cti"
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
                        print("Available components in payload:")
                        for i, component in enumerate(payload.components):
                            print(f"Component {i}: width={component.width}, height={component.height}, data size={component.data.size}")

                        texture_component = payload.components[0]
                        display_texture_if_available(texture_component)
                        display_color_image_with_detection(texture_component, "TextureRGB")

                        point_cloud_component = payload.components[2]
                        point_cloud = create_point_cloud_from_component(point_cloud_component)
                        display_and_detect_objects_in_point_cloud(point_cloud)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Exiting capture loop.")
                            break
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                ia.stop()

# Run the software trigger function
if __name__ == "__main__":
    software_trigger()
