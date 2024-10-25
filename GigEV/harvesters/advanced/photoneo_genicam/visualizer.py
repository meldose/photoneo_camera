from typing import List

import cv2
import glfw
import numpy as np
import open3d as o3d
from harvesters.core import Component2DImage
from open3d.visualization import VisualizerWithKeyCallback

#  defining the function called render static
def render_static(
    objects: List, axes_size=70, width=1000, height=800, top=300, left=300
):
    """
    Render a static Open3D geometry with axes.
    """
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axes_size, origin=np.array([0, 0, 0])
    )
    # visualizing the geometries of the image having width , height ,top and left 
    o3d.visualization.draw_geometries(
        geometry_list=[axes] + objects, width=width, height=height, top=top, left=left
    )


def process_for_visualisation(image: Component2DImage):
    isRGB: bool = image.data_format == "RGB8"
    isDepth: bool = image.data_format == "Coord3D_C32f"
    isConfidence: bool = image.data_format == "Confidence8"
    isMono10: bool = image.data_format == "Mono10"
    isMono12: bool = image.data_format == "Mono12"
    isMono16: bool = image.data_format == "Mono16"
    isNormal: bool = image.data_format == "Coord3D_ABC32f"

    if isMono10:
        upscaled_to_16bit = image.data.astype(np.uint16) << 6
        return upscaled_to_16bit.reshape(image.height, image.width, 1).copy() # reshaping the image into 3D array
    if isMono12:
        upscaled_to_16bit = image.data.astype(np.uint16) << 4
        return upscaled_to_16bit.reshape(image.height, image.width, 1).copy() # reshaping the image into 3D array
    if isConfidence or isMono16:
        return image.data.reshape(image.height, image.width, 1).copy() # reshaping the image into 3D array.
    if isDepth:
        image_array = image.data.reshape((image.height, image.width, 1)) # if it is Depth then reshape the size into 3D array.
        normalized_image = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX)
        return normalized_image.astype(np.uint16)
    if isRGB:
        return cv2.cvtColor(image.data.reshape((image.height, image.width, 3)), cv2.COLOR_RGB2BGR) # if it is RGB , then get the cvtColor and reshape the into 2D array having 3 colors
    if isNormal:
        image_array = image.data.reshape((image.height, image.width, 3)) # reshaping the image into 3D array having 3 colors
        normalized_image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image.astype(np.uint8)
    else:
        raise Exception("Unexpected pixel format") # raise unexcpected pixel format


class TextureImage: # define an class callled TextureImage
    def __init__(self, name: str, image: Component2DImage): # initialize the class with name and image
        self.name = name
        self.image: Component2DImage = image
        self.processed_image = process_for_visualisation(self.image)

    def show(self): # define the function for show
        cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(self.name, self.processed_image) # show the image 

    def print_info(self):
        print(f"Resolution:{self.image.width}x{self.image.height}")
        print(f"ImagePixelFormat:{self.image.data_format}")
        print(f"Components per pixel:{self.image.num_components_per_pixel}")
        print(f"Min:{self.image.data.min()}")
        print(f"Max:{self.image.data.max()}")


class RealTimePCLRenderer: # define the class RealTimePCLRenderer
    def __init__(self): # initialize the class
        self.vis = VisualizerWithKeyCallback()
        self.vis.create_window(width=1000, height=800, top=300, left=100)
        self.render_opts = self.vis.get_render_option()
        self.render_opts.point_size = 2.0
        self.render_opts.background_color = np.asarray([0.9, 0.9, 0.9])
        self.vis.register_key_action_callback(glfw.KEY_ESCAPE, self.key_action_callback)
        self.should_close = False
        print("Window created - OK")

    def key_action_callback(self, vis, key, action): # define the function called key action callback
        self.should_close = True
