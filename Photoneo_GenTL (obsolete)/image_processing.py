import numpy as np
import open3d as o3d
import cv2
import os
import sys
import torch
import argparse
import logging
from harvesters.core import Harvester
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sys import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")  # Optional: Logs to a file named app.log
    ]
)
logger = logging.getLogger(__name__)

# Load YOLO model
try:
    logger.info("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logger.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLOv5 model: {e}")
    sys.exit(1)

# Object classes for specific objects of interest
CLASS_NAMES = ["person", "car"]

def get_texture_image(texture_component):
    """
    Process and retrieve the texture image from the texture component.
    """
    if texture_component.width == 0 or texture_component.height == 0:
        logger.warning("Texture component is empty or unavailable!")
        return None

    try:
        logger.debug(f"Texture component data size: {texture_component.data.size}")
        logger.debug(f"Texture component data type: {texture_component.data.dtype}")

        # Check if data can be reshaped into an image
        expected_size = texture_component.height * texture_component.width * 3  # Assuming 3 channels (RGB)
        if texture_component.data.size != expected_size:
            logger.error(f"Unexpected data size. Expected {expected_size}, got {texture_component.data.size}")
            return None

        # Reshape the data
        texture_image = texture_component.data.reshape(texture_component.height, texture_component.width, 3)
        logger.debug(f"Texture image shape after reshaping: {texture_image.shape}")

        # Normalize and convert to 8-bit if necessary
        if texture_image.dtype != np.uint8:
            texture_image = cv2.normalize(texture_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            logger.debug(f"Texture image data type after normalization: {texture_image.dtype}")

        # Convert color format if needed (OpenCV uses BGR by default)
        if texture_component.data_format.upper() == 'RGB8':
            texture_image = cv2.cvtColor(texture_image, cv2.COLOR_RGB2BGR)

        return texture_image

    except Exception as e:
        logger.error(f"Error processing texture image: {e}")
        return None

def display_color_image_with_detection(color_image, window_name):
    """
    Display the color image with YOLOv5 object detection overlays.
    """
    if color_image is None or color_image.size == 0:
        logger.warning(f"{window_name} is empty!")
        return

    try:
        image_copy = color_image.copy()  # Preserve original image

        # Run YOLO detection on the image
        results = model(image_copy)
        detections = results.pandas().xyxy[0]
        detections = detections[detections['name'].isin(CLASS_NAMES)]

        # Display bounding boxes and labels
        for idx, row in detections.iterrows():
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            width = x_max - x_min
            height = y_max - y_min
            label = f"{row['name']} {row['confidence']:.2f}"
            dimensions_label = f"W:{width}px H:{height}px"

            # Draw rectangle and labels on the copied image
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_copy, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            cv2.putText(image_copy, dimensions_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            logger.info(f"Detected {label} at {(x_min, y_min)} with dimensions {width}px x {height}px")

        # Display the image using OpenCV
        cv2.imshow(window_name, image_copy)

    except Exception as e:
        logger.error(f"Error during object detection: {e}")

def create_point_cloud_from_component(point_cloud_component):
    """
    Create an Open3D point cloud from the point cloud component.
    """
    if point_cloud_component.data.size == 0:
        logger.warning("Point cloud component is empty!")
        return None

    try:
        # Ensure the data is in the expected format
        points = np.array(point_cloud_component.data, copy=False)
        num_points = points.size // 3
        if points.size % 3 != 0:
            logger.error(f"Point cloud data size is not a multiple of 3: {points.size}")
            return None

        points = points.reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        logger.debug("Point cloud created successfully.")

        return point_cloud

    except Exception as e:
        logger.error(f"Error creating point cloud: {e}")
        return None

def display_and_detect_objects_in_point_cloud(point_cloud, open3d_visualizer):
    """
    Process the point cloud to detect clusters and display with bounding boxes.
    """
    if point_cloud is None or len(point_cloud.points) == 0:
        logger.warning("Point cloud is empty or has no points.")
        return

    try:
        # Clustering parameters
        eps = 0.02  # Distance threshold
        min_points = 10  # Minimum number of points to form a cluster

        logger.info("Clustering point cloud using DBSCAN...")
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if labels.size == 0:
            logger.warning("No clusters detected in point cloud.")
            return

        max_label = labels.max()
        logger.info(f"Point cloud has {max_label + 1} clusters.")

        # Assign colors to each cluster
        colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
        colors[labels < 0] = 0  # Noise points
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Update the visualizer
        open3d_visualizer.clear_geometries()
        open3d_visualizer.add_geometry(point_cloud)
        open3d_visualizer.poll_events()
        open3d_visualizer.update_renderer()

        # Calculate and display bounding boxes
        for i in range(max_label + 1):
            cluster = point_cloud.select_by_index(np.where(labels == i)[0])
            bbox = cluster.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # Red bounding box
            open3d_visualizer.add_geometry(bbox)

            size = bbox.get_extent()
            logger.info(f"Cluster {i}: Size (X: {size[0]:.2f}, Y: {size[1]:.2f}, Z: {size[2]:.2f})")

        open3d_visualizer.poll_events()
        open3d_visualizer.update_renderer()

    except Exception as e:
        logger.error(f"Error in point cloud processing: {e}")

def initialize_open3d_visualizer():
    """
    Initialize the Open3D visualizer for real-time point cloud display.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud', width=1024, height=768)
    logger.info("Open3D visualizer initialized.")
    return vis

def software_trigger(cti_file_path, device_id):
    """
    Set up the camera, configure settings, and manage the capture loop.
    """
    logger.info(f"Device ID: {device_id}")
    logger.info(f"CTI File Path: {cti_file_path}")

    if not os.path.isfile(cti_file_path):
        logger.error(f"CTI file not found at {cti_file_path}. Please ensure that the path is correct.")
        sys.exit(1)

    with Harvester() as h:
        try:
            h.add_file(cti_file_path, check_validity=True)
            h.update()

            if not h.device_info_list:
                logger.error("No devices found.")
                sys.exit(1)

            logger.info("\nAvailable devices:")
            logger.info("Serial Number : ID")
            logger.info("---------")
            device_ids = []
            for item in h.device_info_list:
                serial = item.property_dict.get('serial_number', 'Unknown')
                dev_id = item.property_dict.get('id_', 'Unknown')
                device_ids.append(dev_id)
                logger.info(f"{serial} : {dev_id}")
            logger.info("")

            if device_id not in device_ids:
                logger.error(f"Device ID '{device_id}' not found among available devices.")
                sys.exit(1)

            with h.create({'id_': device_id}) as ia:
                features = ia.remote_device.node_map

                # Enable necessary features
                for feature in ['SendTexture', 'SendPointCloud', 'SendNormalMap', 'SendDepthMap', 'SendConfidenceMap']:
                    if hasattr(features, feature):
                        setattr(features, feature, True)
                        logger.info(f"Enabled feature: {feature}")
                    else:
                        logger.warning(f"{feature} feature not available on this device.")

                # Set trigger mode to software
                if hasattr(features, 'PhotoneoTriggerMode'):
                    logger.info(f"TriggerMode BEFORE: {features.PhotoneoTriggerMode.value}")
                    features.PhotoneoTriggerMode.value = "Software"
                    logger.info(f"TriggerMode AFTER: {features.PhotoneoTriggerMode.value}")
                else:
                    logger.warning("PhotoneoTriggerMode feature not available on this device.")

                # Start the camera interface
                ia.start()
                logger.info("Camera interface started.")

                # Initialize Open3D visualizer
                vis = initialize_open3d_visualizer()

                try:
                    while True:
                        logger.info("\n-- Capturing frame --")
                        features.TriggerFrame.execute()

                        with ia.fetch(timeout=10.0) as buffer:
                            payload = buffer.payload
                            logger.debug("Fetched new frame.")

                            logger.info("Available components in payload:")
                            for i, component in enumerate(payload.components):
                                logger.info(f"Component {i}: width={component.width}, height={component.height}, "
                                            f"data size={component.data.size}, data format={component.data_format}")

                            # Process Texture Component
                            texture_component = None
                            for component in payload.components:
                                if component.data_format.upper() in ['RGB8', 'BGR8']:
                                    texture_component = component
                                    logger.info(f"Texture component found with data format: {component.data_format}")
                                    break

                            if texture_component:
                                texture_image = get_texture_image(texture_component)
                                if texture_image is not None:
                                    # Save the texture image
                                    cv2.imwrite("TextureRGB.png", texture_image)
                                    logger.info("Texture RGB image saved as TextureRGB.png")

                                    # Display with object detection
                                    display_color_image_with_detection(texture_image, "TextureRGB")
                            else:
                                logger.warning("Texture component not found in payload components.")

                            # Process Point Cloud Component
                            point_cloud_component = None
                            for component in payload.components:
                                if component.data_format.upper() == 'COORD3D_ABC32F':
                                    point_cloud_component = component
                                    logger.info("Point cloud component found.")
                                    break

                            if point_cloud_component:
                                point_cloud = create_point_cloud_from_component(point_cloud_component)
                                display_and_detect_objects_in_point_cloud(point_cloud, vis)
                            else:
                                logger.warning("Point cloud component not found in payload components.")

                        # Handle key events
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Exiting capture loop.")
                            break

                except KeyboardInterrupt:
                    logger.info("Capture loop interrupted by user.")
                except Exception as e:
                    logger.error(f"An error occurred: {type(e).__name__}: {e}")
                finally:
                    ia.stop()
                    vis.destroy_window()
                    cv2.destroyAllWindows()
                    logger.info("Camera interface stopped and resources released.")

        except Exception as e:
            logger.error(f"An error occurred during camera setup: {type(e).__name__}: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Photoneo Camera Capture and Processing")
    parser.add_argument('--device_id', type=str, default="TER-008", help='Device ID of the Photoneo camera')
    parser.add_argument('--cti_path', type=str, help='Path to the photoneo.cti file')
    args = parser.parse_args()

    # Determine the CTI file path
    if args.cti_path:
        cti_file_path = args.cti_path
    else:
        # Check if PHOXI_CONTROL_PATH is set
        cti_base_path = os.getenv('PHOXI_CONTROL_PATH')
        if not cti_base_path:
            logger.error("Environment variable 'PHOXI_CONTROL_PATH' is not set and --cti_path was not provided.")
            sys.exit(1)
        
        # Construct the default CTI file path
        cti_file_path = os.path.join(cti_base_path, "API/lib/photoneo.cti")

    # Verify that the CTI file exists
    if not os.path.isfile(cti_file_path):
        logger.error(f"CTI file not found at {cti_file_path}. Please ensure that the path is correct.")
        sys.exit(1)

    software_trigger(cti_file_path, args.device_id)

if __name__ == "__main__":
    main()
