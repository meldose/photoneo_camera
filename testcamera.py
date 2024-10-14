import logging
import keyboard  # Library for detecting keypresses
from phoXi import PhoXiControl, PhoXiError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def capture_image(phoXi_device):
    """Function to capture an image and process the point cloud."""
    logging.info("Capturing image...")
    frame = phoXi_device.TriggerFrame()

    if frame:
        logging.info("Image captured successfully.")

        # Access point cloud data from the captured frame
        point_cloud = frame.GetPointCloud()
        logging.info(f"Point cloud has {len(point_cloud)} points.")

        # Example: Save point cloud to a PLY file
        save_path = "captured_point_cloud.ply"
        success = frame.SaveAsPLY(save_path)
        if success:
            logging.info(f"Point cloud saved to {save_path}.")
        else:
            logging.error("Failed to save point cloud.")
    else:
        logging.error("Failed to capture image.")


def main():
    try:
        # Initialize the PhoXi Control Interface
        phoXi = PhoXiControl()
        logging.info("PhoXi Control Interface initialized.")

        # Get a list of available devices
        devices = phoXi.GetDeviceList()

        if not devices:
            logging.error("No PhoXi device found.")
            return

        # List available devices
        logging.info("Available PhoXi devices:")
        for idx, device in enumerate(devices):
            logging.info(f"Device {idx}: {device.HWIdentification}, Type: {device.Type}")

        # User selects a device
        if len(devices) > 1:
            selected_index = int(input(f"Select device (0-{len(devices)-1}): "))
            if selected_index < 0 or selected_index >= len(devices):
                logging.error("Invalid device index selected.")
                return
            device = devices[selected_index]
        else:
            device = devices[0]

        logging.info(f"Connecting to device {device.HWIdentification}...")

        # Connect to the selected device
        phoXi_device = phoXi.CreateAndConnect(device.HWIdentification)

        if not phoXi_device:
            logging.error("Failed to connect to the device.")
            return

        logging.info("Device connected successfully.")

        # Ensure acquisition is stopped before starting
        if phoXi_device.isAcquiring():
            phoXi_device.StopAcquisition()
            logging.info("Existing acquisition stopped.")

        # Set trigger mode to software
        phoXi_device.TriggerMode = "Software"
        logging.info("Trigger mode set to Software.")

        # Start acquisition
        phoXi_device.StartAcquisition()
        logging.info("Acquisition started.")

        logging.info("Press 't' to trigger the camera capture, or 'q' to quit.")

        # Main loop to wait for key presses
        while True:
            if keyboard.is_pressed('t'):  # Trigger capture on 't' key press
                logging.info("'t' key pressed. Triggering camera...")
                capture_image(phoXi_device)

            if keyboard.is_pressed('q'):  # Quit loop on 'q' key press
                logging.info("'q' key pressed. Exiting...")
                break

    except PhoXiError as e:
        logging.exception(f"PhoXiError encountered: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        # Stop acquisition and disconnect device if connected
        try:
            if 'phoXi_device' in locals():
                if phoXi_device.isAcquiring():
                    phoXi_device.StopAcquisition()
                    logging.info("Acquisition stopped.")
                phoXi_device.Disconnect()
                logging.info("Device disconnected.")
        except Exception as e:
            logging.exception(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()
