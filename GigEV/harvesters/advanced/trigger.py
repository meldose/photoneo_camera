import sys
import time
from pathlib import Path
from typing import List

from genicam.genapi import NodeMap
from harvesters.core import Component2DImage, Harvester
from photoneo_genicam.default_gentl_producer import producer_path
from photoneo_genicam.features import enable_hardware_trigger
from photoneo_genicam.user_set import load_default_user_set


def main(device_sn: str, num_frames: int = 100):  # Adding num_frames parameter for FPS calculation
    """Connects to the device using the provided serial number and fetches images using hardware trigger."""
    with Harvester() as h:
        h.add_file(str(producer_path), check_existence=True, check_validity=True)
        h.update()  # Update harvester

        print(f"Connecting to: {device_sn}")  # Connect to device
        try:
            with h.create({"serial_number": device_sn}) as image_acquirer:  # Connect to device with device_sn
                features: NodeMap = image_acquirer.remote_device.node_map  # Get node map

                load_default_user_set(features)
                enable_hardware_trigger(features)

                image_acquirer.start()  # Start the device   

                # Initialize variables for FPS calculation
                frame_count = 0
                start_time = time.time()
                
                while frame_count < num_frames:
                    # Here you can implement the hardware trigger. This may depend on your specific setup.
                    # For example, if you are using a GPIO library, it might look something like this:
                    # trigger_hardware_signal()

                    timeout = 5  # Adjust the timeout as needed for your hardware
                    print(f"Waiting for hw-trigger signal... (Timeout: {timeout}s)")
                    
                    with image_acquirer.fetch(timeout=timeout) as buffer:  # Fetch the buffer with timeout
                        component_list: List[Component2DImage] = buffer.payload.components
                        for component in component_list:  # Process each component
                            print(f"Captured Frame {frame_count + 1}: {component}")  # Print out the component
                            frame_count += 1

                    # Break if we've captured enough frames
                    if frame_count >= num_frames:
                        break

                # Calculate FPS
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"Captured {frame_count} frames in {elapsed_time:.2f} seconds.")
                print(f"Frames Per Second (FPS): {fps:.2f}")

        except Exception as e:
            print(f"Error during device operation: {e}")  # Print error if device operation fails


if __name__ == "__main__":
    try:
        device_id = sys.argv[1]
        print(f"Device ID: {device_id}")  # Debug statement
    except IndexError:
        print("Error: no device given, please run it with the device serial number as argument:")
        print(f"    {Path(__file__).name} <device serial>")  # Print the path file with device serial
        sys.exit(1)

    # Check if device_id is valid
    if not device_id:  # If the device_id is None or an empty string
        print("Error: Device ID cannot be None or empty.")  # Print the error
        sys.exit(1)

    main(device_id)  # Call the function main()
