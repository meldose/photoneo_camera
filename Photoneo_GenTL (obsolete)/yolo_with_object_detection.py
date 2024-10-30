
# import torch
# import cv2

# # Load YOLO model (e.g., YOLOv5 pre-trained model)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Define the target classes for detection (e.g., 'person', 'car')
# target_classes = ['person', 'car']  # Replace with your specific classes

# # Initialize Camera (OpenCV for a USB webcam example)
# camera = cv2.VideoCapture(0)  # '0' is typically the default camera, change if needed

# # Check if the camera opened successfully
# if not camera.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Define a function to process detections and check for target objects
# def process_detections(results):
#     for detection in results:
#         if detection['name'] in target_classes:
#             return True  # Trigger camera
#     return False

# # Main Detection Loop
# def main():
#     while True:
#         # Capture frame from the camera
#         ret, frame = camera.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Run YOLO inference on the frame
#         results = model(frame)

#         # Process detections to check for target objects
#         detected = process_detections(results.pandas().xyxy[0].to_dict(orient="records"))

#         # If a target object is detected, print a trigger message
#         if detected:
#             print("Target object detected! Triggering camera...")
#             # Placeholder for camera-specific trigger function
#             # e.g., camera.trigger() for Photoneo or GigE Vision SDK

#         # Draw bounding boxes for all detections on the frame
#         for idx, row in results.pandas().xyxy[0].iterrows():
#             x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             label = f"{row['name']} {row['confidence']:.2f}"

#             # Draw the bounding box and label
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Display the frame with bounding boxes
#         cv2.imshow("YOLO Object Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     camera.release()
#     cv2.destroyAllWindows()

# # Run main loop
# if __name__ == "__main__":
#     main()


import torch
import cv2

# Load YOLO model (e.g., YOLOv5 pre-trained model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the target classes for detection (e.g., 'person', 'car')
target_classes = ['person', 'car']  # Replace with your specific classes

# Initialize Camera (OpenCV for a USB webcam example)
camera = cv2.VideoCapture(0)  # '0' is typically the default camera, change if needed

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define a function to process detections and check for target objects
def process_detections(results):
    for detection in results:
        if detection['name'] in target_classes:
            return True  # Trigger camera
    return False

# Main Detection Loop
def main():
    while True:
        # Capture frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Process detections to check for target objects
        detected = process_detections(results.pandas().xyxy[0].to_dict(orient="records"))

        # If a target object is detected, print a trigger message
        if detected:
            print("Target object detected! Triggering camera...")
            # Placeholder for camera-specific trigger function
            # e.g., camera.trigger() for Photoneo or GigE Vision SDK

        # Draw bounding boxes for all detections on the frame
        for idx, row in results.pandas().xyxy[0].iterrows():
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"

            # Calculate width and height of the detected object
            width = x_max - x_min
            height = y_max - y_min
            dimensions_label = f"W:{width}px H:{height}px"

            # Draw the bounding box and labels (name, confidence, dimensions)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, dimensions_label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes and dimensions
        cv2.imshow("YOLO Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    camera.release()
    cv2.destroyAllWindows()

# Run main loop
if __name__ == "__main__":
    main()
