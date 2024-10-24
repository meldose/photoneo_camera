import sys
from pathlib import Path
from typing import List

from genicam.genapi import NodeMap
from harvesters.core import Component2DImage, Harvester

from photoneo_genicam.default_gentl_producer import producer_path
from photoneo_genicam.features import enable_hardware_trigger
from photoneo_genicam.user_set import load_default_user_set


def main(device_sn: str): # define the main function
    with Harvester() as h:
        h.add_file(str(producer_path), check_existence=True, check_validity=True)
        h.update() # update harvester

        print(f"Connecting to: {device_sn}") # connect to device
        with h.create({"serial_number": device_sn}) as ia: # connect to device with device_sn
            features: NodeMap = ia.remote_device.node_map # get node map

            load_default_user_set(features)
            enable_hardware_trigger(features)

            ia.start() # start the device   
            timeout = 180 # consider the timeout as 180
            print(f"Wait {timeout}s for hw-trigger signal...")
            with ia.fetch(timeout=timeout) as buffer: # fetch the buffer with timeout
                component_list: List[Component2DImage] = buffer.payload.components
                for component in component_list: # conisder the components in component_list
                    print(component) # print out the component


if __name__ == "__main__":
    try:
        device_id = sys.argv[1]
        print(f"Device ID: {device_id}")  # Debug statement
    except IndexError:
        print("Error: no device given, please run it with the device serial number as argument:")
        print(f"    {Path(__file__).name} <device serial>") # print the path file with device serial
        sys.exit(1)

    # Check if device_id is valid
    if device_id is None: # if the device_id is None
        print("Error: Device ID cannot be None.") # print the error
        sys.exit(1)

    main(device_id)# calling the function main()






