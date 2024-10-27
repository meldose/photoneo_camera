#!/usr/bin/env python3
import sys # imported sys module
import os # imported os module
from pathlib import Path # imported Path module

import open3d as o3d # imported module o3d
from genicam.genapi import NodeMap # imported module NodeMap
from harvesters.core import Component2DImage, Harvester # imported module Harvester and Component2DImage from harvesters.core
from photoneo_genicam.components import enable_components # imported function enable_components
from photoneo_genicam.default_gentl_producer import producer_path # imported module producer_path
from photoneo_genicam.features import enable_software_trigger # imported function enable_software_trigger
from photoneo_genicam.pointcloud import create_3d_vector # imported function create_3d_vector
from photoneo_genicam.user_set import load_default_user_set # imported function load_default_user_set
from photoneo_genicam.visualizer import render_static # imported function render_static
 

def check_environment(): # defined function check_environment
    """Check if the GENICAM_GENTL64_PATH is set."""
    gentl_path = os.getenv("GENICAM_GENTL64_PATH") # set the variable gentl_path to the value of the environment variable GENICAM_GENTL64_PATH
    if not gentl_path: # if the variable gentl_path is not set
        raise EnvironmentError("GENICAM_GENTL64_PATH is not set. " # raise an EnvironmentError
                               "Please set it to the path containing GenTL producer libraries.")
    print(f"GENICAM_GENTL64_PATH is set to: {gentl_path}")


def main(device_sn: str): # defined function main having device_sn as an argument
    """Capture a point cloud from the specified device."""
    check_environment() # call the check_environment function
    
    with Harvester() as h: # consider h as Harvester
        h.add_file(str(producer_path), check_existence=True, check_validity=True) # add file to harvester
        h.update() # update harvester

        with h.create({"serial_number": device_sn}) as ia: # connect to device
            features: NodeMap = ia.remote_device.node_map
            
            load_default_user_set(features) # load default user set
            enable_software_trigger(features) # enable software trigger
            enable_components(features, ["Range"]) # enable components with Range
            features.Scan3dOutputMode.value = "CalibratedABC_Grid" # set Scan3dOutputMode as CalibratedABC_Grid

            ia.start() # start the device
            features.TriggerSoftware.execute() # execute the software trigger
            
            try:
                with ia.fetch() as buffer: # fetch the buffer
                    point_cloud_raw: Component2DImage = buffer.payload.components[0] # consider the point_cloud_raw as Component2DImage with Range
                    point_cloud = o3d.geometry.PointCloud() # create a point cloud
                    point_cloud.points = create_3d_vector(point_cloud_raw.data.copy()) # set points
                    o3d.io.write_point_cloud("pointcloud.ply", point_cloud) # write the point cloud
                    render_static([point_cloud]) # render the point cloud
            except Exception as e:
                print(f"Error fetching point cloud: {e}") # print the error
                sys.exit(1)


if __name__ == "__main__": # checking if the name of the file is main
    try:
        device_id = sys.argv[1]
    except IndexError: # exception IndexError happens when there is no argument
        print("Error: No device given. Please run it with the device serial number as an argument:") # print error message
        print(f"    {Path(__file__).name} <device serial>") # print the path file
        sys.exit(1)

main(device_id) # calling the main function having device_id
