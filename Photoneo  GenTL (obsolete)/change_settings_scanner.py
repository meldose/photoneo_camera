import os # imported os module
import sys # imported sys module
from sys import platform # imported platform module
from harvesters.core import Harvester # imported module Harvester
import struct # imported struct module

# PhotoneoTL_DEV_<ID>

#################### # checking if the length of the argument is 2
device_id = "PhotoneoTL_DEV_TER-008"# assigning the device id to the argument value
print("--> device_id: ", device_id) # printing the device id

if platform == "linux": # if the platform is linux
    cti_file_path_suffix = "/API/bin/photoneo.cti" # set the cti file path suffix as follows
else:
    cti_file_path_suffix = "/API/lib/photoneo.cti"
cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
print("--> cti_file_path: ", cti_file_path) # printing the cti file path

with Harvester() as h: # consider h as Harvester
    h.add_file(cti_file_path, True, True) # adding the file path
    h.update() # updating the harvester

    # Print out available devices
    print()
    print("Name : ID")
    print("---------")
    for item in h.device_info_list: # checking if the item is in the info list or not
        print(item.property_dict['serial_number'], ' : ', item.property_dict['id_']) #  printing the serial_number with item property dict
    print()

    with h.create({'id_': device_id}) as ia: # creating the device id
        features = ia.remote_device.node_map # assigning the features as ia.remote_device.node_map

        ## General settings
        # ReadOnly
        is_phoxi_control_running = features.IsPhoXiControlRunning.value # assigning the is_phoxi_control_running as features.IsPhoXiControlRunning.value
        api_version = features.PhotoneoAPIVersion.value # assigning the api_version as features.PhotoneoAPIVersion.value
        id = features.PhotoneoDeviceID.value # assigning the id as features.PhotoneoDeviceID.value
        type = features.PhotoneoDeviceType.value # assigning the type as features.PhotoneoDeviceType.value
        is_acquiring = features.IsAcquiring.value # assigning the is_acquiring as features.IsAcquiring.value
        is_connected = features.IsConnected.value # assigning the is_connected as features.IsConnected.value
        device_firmware_version = features.PhotoneoDeviceFirmwareVersion.value # assigning the device_firmware_version as features.PhotoneoDeviceFirmwareVersion.value
        device_variant = features.PhotoneoDeviceVariant.value # assigning the device_variant as features.PhotoneoDeviceVariant.value
        device_features = features.PhotoneoDeviceFeatures.value # assigning the device_features as features.PhotoneoDeviceFeatures.value

        if type != "PhoXi3DScanner": # checking if the type is not equal to PhoXi3DScanner
            print("Device is not a PhoXi3DScanner!") # printing the device is not a PhoXi3DScanner
            sys.exit(0)

        # `Freerun` or `Software`
        trigger_mode = features.PhotoneoTriggerMode.value # assigning the trigger_mode as features.PhotoneoTriggerMode.value
        features.PhotoneoTriggerMode.value = 'Freerun' # assigning the trigger_mode as features.PhotoneoTriggerMode.value
        # True or False
        wait_for_grabbing_end = features.WaitForGrabbingEnd.value # assigning the wait_for_grabbing_end as features.WaitForGrabbingEnd.value
        features.WaitForGrabbingEnd.value = False # setting the wait_for_grabbing_end as False
        # Timeout in ms, or values 0 (ZeroTimeout) and -1 (Infinity)
        get_frame_timeout = features.GetFrameTimeout.value
        features.GetFrameTimeout.value = 5000 #setting the get_frame_timeout as 5000
        # True or False
        logout_after_disconnect = features.LogoutAfterDisconnect.value # assigning the logout_after_disconnect as features.LogoutAfterDisconnect.value
        features.LogoutAfterDisconnect.value = False # setting the logout_after_disconnect as False
        # True or False
        stop_acquisition_after_disconnect = features.StopAcquisitionAfterDisconnect.value # assigning the stop_acquisition_after_disconnect as features.StopAcquisitionAfterDisconnect.value
        features.StopAcquisitionAfterDisconnect.value = False # setting the stop_acquisition_after_disconnect as False


        ## Capturing settings
        # <1, 20>
        shutter_multiplier = features.ShutterMultiplier.value # assigning the shutter_multiplier as features.ShutterMultiplier.value
        features.ShutterMultiplier.value = 5 # setting the shutter_multiplier as 5
        # <1, 20>
        scan_multiplier = features.ScanMultiplier.value # assigning the scan_multiplier as features.ScanMultiplier.value
        features.ScanMultiplier.value = 5 # setting the scan_multiplier as 5
        # `Res_2064_1544` or `Res_1032_772`
        resolution = features.Resolution.value # assigning the resolution as features.Resolution.value
        features.Resolution.value = 'Res_1032_772' #setting the resolution as Res_1032_772
        # True or False
        camera_only_mode = features.CameraOnlyMode.value # assigning the camera_only_mode as features.CameraOnlyMode.value
        features.CameraOnlyMode.value = False # setting the camera_only_mode as False
        # True or False
        ambient_light_suppression = features.AmbientLightSuppression.value # assigning the ambient_light_suppression as features.AmbientLightSuppression.value
        features.AmbientLightSuppression.value = False #setting the ambient_light_suppression as False
        # `Normal` or `Interreflections`
        coding_strategy = features.CodingStrategy.value # assigning the coding_strategy as features.CodingStrategy.value
        features.CodingStrategy.value = 'Normal' #setting the coding_strategy as Normal
        # `Fast`, `High` or `Ultra`
        coding_quality = features.CodingQuality.value # assigning the coding_quality as features.CodingQuality.value
        features.CodingQuality.value = 'Ultra' #setting the coding_quality as Ultra
        # `Computed`, `LED`, `Laser` or `Focus`
        texture_source = features.TextureSource.value # assigning the texture_source as features.TextureSource.value
        features.TextureSource.value = 'Laser' #setting the texture_source as Laser
        # <10.24, 100.352>
        single_pattern_exposure = features.SinglePatternExposure.value # assigning the single_pattern_exposure as features.SinglePatternExposure.value
        features.SinglePatternExposure.value = 10.24 #setting the single_pattern_exposure as 10.24
        # <0.0, 100.0>
        maximum_fps = features.MaximumFPS.value # assigning the maximum_fps as features.MaximumFPS.value
        features.MaximumFPS.value = 25 #setting the maximum_fps as 25
        # <1, 4095>
        laser_power = features.LaserPower.value # assigning the laser_power as features.LaserPower.value
        features.LaserPower.value = 2000 #setting the laser_power as 2000
        # <0, 512>
        projection_offset_left = features.ProjectionOffsetLeft.value # assigning the projection_offset_left as features.ProjectionOffsetLeft.value
        features.ProjectionOffsetLeft.value = 50 #setting the projection_offset_left as 50
        # <0, 512>
        projection_offset_right = features.ProjectionOffsetRight.value # assigning the projection_offset_right as features.ProjectionOffsetRight.value
        features.ProjectionOffsetRight.value = 50 #setting the projection_offset_right as 50
        # <1, 4095>
        led_power = features.LEDPower.value # assigning the led_power as features.LEDPower.value
        features.LEDPower.value = 2000 #setting the led_power as 2000
        # True or False
        hardware_trigger = features.HardwareTrigger.value # assigning the hardware_trigger as features.HardwareTrigger.value
        features.HardwareTrigger.value = True #setting the hardware_trigger as True
        # `Falling`, `Rising` or `Both`
        hardware_trigger_signal = features.HardwareTriggerSignal.value # assigning the hardware_trigger_signal as features.HardwareTriggerSignal.value
        features.HardwareTriggerSignal.value = 'Both' #setting the hardware_trigger_signal as Both


        ## Processing settings
        # <0.0, 100.0>
        max_inaccuracy = features.MaxInaccuracy.value # assigning the max_inaccuracy as features.MaxInaccuracy.value
        features.MaxInaccuracy.value = 3.5 #setting the max_inaccuracy as 3.5
        # `MinX`, `MinY`, `MinZ`, `MaxX`, `MaxY` or `MaxZ`
        camera_space_selector = features.CameraSpaceSelector.value # assigning the camera_space_selector as features.CameraSpaceSelector.value
        features.CameraSpaceSelector.value = 'MinZ' #setting the camera_space_selector as MinZ
        # <-999999.0, 999999.0>
        camera_space_value = features.CameraSpaceValue.value # assigning the camera_space_value as features.CameraSpaceValue.value
        features.CameraSpaceValue.value = 100.5 #setting the camera_space_value as 100.5
        # `MinX`, `MinY`, `MinZ`, `MaxX`, `MaxY` or `MaxZ`
        point_cloud_space_selector = features.PointCloudSpaceSelector.value # assigning the point_cloud_space_selector as features.PointCloudSpaceSelector.value
        features.PointCloudSpaceSelector.value = 'MaxY' #setting the point_cloud_space_selector as MaxY
        # <-999999.0, 999999.0>
        point_cloud_space_value = features.PointCloudSpaceValue.value # assigning the point_cloud_space_value as features.PointCloudSpaceValue.value
        features.PointCloudSpaceValue.value = 200.5 #   setting the point_cloud_space_value as 200.5
        # <0.0, 90.0>
        max_camera_angle = features.MaxCameraAngle.value # assigning the max_camera_angle as features.MaxCameraAngle.value
        features.MaxCameraAngle.value = 10 #setting the max_camera_angle as 10
        # <0.0, 90.0>
        max_projector_angle = features.MaxProjectorAngle.value # assigning the max_projector_angle as features.MaxProjectorAngle.value
        features.MaxProjectorAngle.value = 15 #setting the max_projector_angle as 15
        # <0.0, 90.0>
        min_halfway_angle = features.MinHalfwayAngle.value # assigning the min_halfway_angle as features.MinHalfwayAngle.value
        features.MinHalfwayAngle.value = 20 #setting the min_halfway_angle as 20
        # <0.0, 90.0> 
        max_halfway_angle = features.MaxHalfwayAngle.value # assigning the max_halfway_angle as features.MaxHalfwayAngle.value
        features.MaxHalfwayAngle.value = 25 #setting the max_halfway_angle as 25
        # True or False
        calibration_volume_only = features.CalibrationVolumeOnly.value # assigning the calibration_volume_only as features.CalibrationVolumeOnly.value
        features.CalibrationVolumeOnly.value = True #setting the calibration_volume_only as True
        # `Sharp`, `Normal` or `Smooth`
        surface_smoothness = features.SurfaceSmoothness.value # assigning the surface_smoothness as features.SurfaceSmoothness.value
        features.SurfaceSmoothness.value = 'Normal' #setting the surface_smoothness as Normal
        # <0, 4> 
        normals_estimation_radius = features.NormalsEstimationRadius.value # assigning the normals_estimation_radius as features.NormalsEstimationRadius.value
        features.NormalsEstimationRadius.value = 1 #setting the normals_estimation_radius as 1
        # True or False
        interreflections_filtering = features.InterreflectionsFiltering.value # assigning the interreflections_filtering as features.InterreflectionsFiltering.value
        features.InterreflectionsFiltering.value = True #setting the interreflections_filtering as True
        # <0.01, 0.99>
        interreflections_filtering_strength = features.InterreflectionFilterStrength.value # assigning the interreflections_filtering_strength as features.InterreflectionFilterStrength.value
        features.InterreflectionFilterStrength.value = 0.50 #setting the interreflections_filtering_strength as 0.50
        # `Local`, `Small`, `Medium` or `Large`
        pattern_decomposition_reach = features.PatternDecompositionReach.value # assigning the pattern_decomposition_reach as features.PatternDecompositionReach.value
        features.PatternDecompositionReach.value = 'Medium' #setting the pattern_decomposition_reach as Medium
        # <0.0, 4095.0>
        signal_contrast_threshold = features.SignalContrastThreshold.value # assigning the signal_contrast_threshold as features.SignalContrastThreshold.value
        features.SignalContrastThreshold.value = 2000.50 #  setting the signal_contrast_threshold as 2000.50
 

        ## Coordinates settings
        # `CameraSpace`, `MarkerSpace`, `RobotSpace` or `CustomSpace`
        camera_space = features.CoordinateSpace.value # assigning the camera_space as features.CoordinateSpace.value
        features.CoordinateSpace.value = 'MarkerSpace' #setting the camera_space as MarkerSpace
        # `Custom` or `Robot`
        transformation_space_selector = features.TransformationSpaceSelector.value # assigning the transformation_space_selector as features.TransformationSpaceSelector.value
        features.TransformationSpaceSelector.value = 'Robot' #setting the transformation_space_selector as Robot
        # `Row0Col0`, `Row0Col1`, `Row0Col2`, `Row1Col0`, .. , `Row2Col2`
        transformation_rotation_matrix_selector = features.TransformationRotationMatrixSelector.value # assigning the transformation_rotation_matrix_selector as features.TransformationRotationMatrixSelector.value
        features.TransformationRotationMatrixSelector.value = 'Row0Col1' #setting the transformation_rotation_matrix_selector as Row0Col1
        # <-999999.0, 999999.0>
        transformation_rotation_matrix_value = features.TransformationRotationMatrixValue.value # assigning the transformation_rotation_matrix_value as features.TransformationRotationMatrixValue.value
        features.TransformationRotationMatrixValue.value = 150.25 #setting the transformation_rotation_matrix_value as 150.25
        # Read/Write as raw bytes array    
        custom_transformation_rotation_matrix_length = features.CustomTransformationRotationMatrix.length # assigning the custom_transformation_rotation_matrix_length as features.CustomTransformationRotationMatrix.length
        custom_transformation_rotation_matrix_bytes = features.CustomTransformationRotationMatrix.get(custom_transformation_rotation_matrix_length) # assigning the custom_transformation_rotation_matrix_bytes as features.CustomTransformationRotationMatrix.get(custom_transformation_rotation_matrix_length)
        custom_transformation_rotation_matrix = struct.unpack('9d', custom_transformation_rotation_matrix_bytes) # assigning the custom_transformation_rotation_matrix as struct.unpack('9d', custom_transformation_rotation_matrix_bytes)
        custom_transformation_rotation_matrix_new_values = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9] #setting the custom_transformation_rotation_matrix_new_values as [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        custom_transformation_rotation_matrix_new_bytes = struct.pack('9d', *custom_transformation_rotation_matrix_new_values) # assigning the custom_transformation_rotation_matrix_new_bytes as struct.pack('9d', *custom_transformation_rotation_matrix_new_values)
        features.CustomTransformationRotationMatrix.set(custom_transformation_rotation_matrix_new_bytes) # setting the custom_transformation_rotation_matrix_new_bytes as features.CustomTransformationRotationMatrix.set(custom_transformation_rotation_matrix_new_bytes)
        robot_transformation_rotation_matrix_length = features.RobotTransformationRotationMatrix.length #   assigning the robot_transformation_rotation_matrix_length as features.RobotTransformationRotationMatrix.length
        robot_transformation_rotation_matrix_bytes = features.RobotTransformationRotationMatrix.get(robot_transformation_rotation_matrix_length) # assigning the robot_transformation_rotation_matrix_bytes as features.RobotTransformationRotationMatrix.get(robot_transformation_rotation_matrix_length)
        robot_transformation_rotation_matrix = struct.unpack('9d', robot_transformation_rotation_matrix_bytes) # assigning the robot_transformation_rotation_matrix as struct.unpack('9d', robot_transformation_rotation_matrix_bytes)
        robot_transformation_rotation_matrix_new_values = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9] #setting the robot_transformation_rotation_matrix_new_values as [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        robot_transformation_rotation_matrix_new_bytes = struct.pack('9d', *robot_transformation_rotation_matrix_new_values) # assigning the robot_transformation_rotation_matrix_new_bytes as struct.pack('9d', *robot_transformation_rotation_matrix_new_values)
        features.RobotTransformationRotationMatrix.set(robot_transformation_rotation_matrix_new_bytes) # setting the robot_transformation_rotation_matrix_new_bytes as features.RobotTransformationRotationMatrix.set(robot_transformation_rotation_matrix_new_bytes)
        # `X`, `Y` or `Z`
        transformation_translation_vector_selector = features.TransformationTranslationVectorSelector.value # assigning the transformation_translation_vector_selector as features.TransformationTranslationVectorSelector.value
        features.TransformationTranslationVectorSelector.value = 'Z' #setting the transformation_translation_vector_selector as Z
        # <-999999.0, 999999.0>
        transformation_translation_vector_value = features.TransformationTranslationVectorValue.value # assigning the transformation_translation_vector_value as features.TransformationTranslationVectorValue.value
        features.TransformationTranslationVectorValue.value = 225.50 # setting the transformation_translation_vector_value as 225.50
        # Read/Write as raw bytes array      
        custom_transformation_translation_vector_length = features.CustomTransformationTranslationVector.length # assigning the custom_transformation_translation_vector_length as features.CustomTransformationTranslationVector.length
        custom_transformation_translation_vector_bytes = features.CustomTransformationTranslationVector.get(custom_transformation_translation_vector_length) # assigning the custom_transformation_translation_vector_bytes as features.CustomTransformationTranslationVector.get(custom_transformation_translation_vector_length)
        custom_transformation_translation_vector = struct.unpack('3d', custom_transformation_translation_vector_bytes) # assigning the custom_transformation_translation_vector as struct.unpack('3d', custom_transformation_translation_vector_bytes)
        custom_transformation_translation_vector_new_values = [1.1, 2.2, 3.3] #setting the custom_transformation_translation_vector_new_values as [1.1, 2.2, 3.3]
        custom_transformation_translation_vector_new_bytes = struct.pack('3d', *custom_transformation_translation_vector_new_values) # assigning the custom_transformation_translation_vector_new_bytes as struct.pack('3d', *custom_transformation_translation_vector_new_values)
        features.CustomTransformationTranslationVector.set(custom_transformation_translation_vector_new_bytes) # setting the custom_transformation_translation_vector_new_bytes as features.CustomTransformationTranslationVector.set(custom_transformation_translation_vector_new_bytes)
        robot_transformation_translation_vector_length = features.RobotTransformationTranslationVector.length
        robot_transformation_translation_vector_bytes = features.RobotTransformationTranslationVector.get(robot_transformation_translation_vector_length)
        robot_transformation_translation_vector = struct.unpack('3d', robot_transformation_translation_vector_bytes) # assigning the robot_transformation_translation_vector as struct.unpack('3d', robot_transformation_translation_vector_bytes)
        robot_transformation_translation_vector_new_values = [1.1, 2.2, 3.3] #setting the robot_transformation_translation_vector_new_values as [1.1, 2.2, 3.3]
        robot_transformation_translation_vector_new_bytes = struct.pack('3d', *robot_transformation_translation_vector_new_values) # assigning the robot_transformation_translation_vector_new_bytes as struct.pack('3d', *robot_transformation_translation_vector_new_values)
        features.RobotTransformationTranslationVector.set(robot_transformation_translation_vector_new_bytes) # setting the robot_transformation_translation_vector_new_bytes as features.RobotTransformationTranslationVector.set(robot_transformation_translation_vector_new_bytes)
        # True or False
        recognize_markers = features.RecognizeMarkers # assigning the recognize_markers as features.RecognizeMarkers
        features.RecognizeMarkers.value = True #setting the recognize_markers as True
        # <-999999.0, 999999.0>
        marker_scale_width = features.MarkerScaleWidth # assigning the marker_scale_width as features.MarkerScaleWidth
        features.MarkerScaleWidth.value = 0.50 #setting the marker_scale_width as 0.50
        # <-999999.0, 999999.0>
        marker_scale_height = features.MarkerScaleHeight # assigning the marker_scale_height as features.MarkerScaleHeight
        features.MarkerScaleHeight.value = 0.50 #setting the marker_scale_height as 0.50


        ## Calibration settings
        # `Row0Col0`, `Row0Col1`, `Row0Col2`, `Row1Col0`, .. , `Row2Col2`
        camera_matrix_selector = features.CameraMatrixSelector.value # assigning the camera_matrix_selector as features.CameraMatrixSelector
        features.CameraMatrixSelector.value = 'Row0Col1' #setting the camera_matrix_selector as Row0Col1
        # ReadOnly
        camera_matrix_value = features.CameraMatrixValue.value # assigning the camera_matrix_value as features.CameraMatrixValue.value
        # Read as raw bytes array
        camera_matrix_length = features.CameraMatrix.length # assigning the camera_matrix_length as features.CameraMatrix.length
        camera_matrix_bytes = features.CameraMatrix.get(camera_matrix_length) # assigning the camera_matrix_bytes as features.CameraMatrix.get(camera_matrix_length)
        camera_matrix = struct.unpack('9d', camera_matrix_bytes) # assigning the camera_matrix as struct.unpack('9d', camera_matrix_bytes)
        # <0, 13>
        distortion_coefficient_selector = features.DistortionCoefficientSelector.value # assigning the distortion_coefficient_selector as features.DistortionCoefficientSelector.value
        features.DistortionCoefficientSelector.value = 3 #setting the distortion_coefficient_selector as 3
        # ReadOnly
        distortion_coefficient_value = features.DistortionCoefficientValue.value # assigning the distortion_coefficient_value as features.DistortionCoefficientValue.value
        # Read as raw bytes array
        distortion_coefficient_length = features.DistortionCoefficient.length # assigning the distortion_coefficient_length as features.DistortionCoefficient.length
        distortion_coefficient_bytes = features.DistortionCoefficient.get(distortion_coefficient_length) # assigning the distortion_coefficient_bytes as features.DistortionCoefficient.get(distortion_coefficient_length)
        distortion_coefficient = struct.unpack('14d', distortion_coefficient_bytes) # assigning the distortion_coefficient as struct.unpack('14d', distortion_coefficient_bytes)

        focus_length = features.FocusLength.value #setting the focus_length as features.FocusLength.value
        pixel_length_width = features.PixelSizeWidth.value #setting the pixel_length_width as features.PixelSizeWidth.value
        pixel_length_height = features.PixelSizeHeight.value #setting the pixel_length_height as features.PixelSizeHeight.value


        ## FrameOutput settings
        # Enable/Disable transfer of spefific images (True or False)
        features.SendPointCloud.value = True #setting the features.SendPointCloud.value as True
        features.SendNormalMap.value = True #setting the features.SendNormalMap.value as True
        features.SendDepthMap.value = True #setting the features.SendDepthMap.value as True
        features.SendConfidenceMap.value = True #setting the features.SendConfidenceMap.value as True
        features.SendTexture.value = True #setting the features.SendTexture.value as True

        # The ia object will automatically call the destroy method
        # once it goes out of the block.

    # The h object will automatically call the reset method
    # once it goes out of the block.
