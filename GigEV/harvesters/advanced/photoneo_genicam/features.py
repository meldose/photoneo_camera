from genicam.genapi import NodeMap # immported the NodeMap module


def enable_trigger(features: NodeMap, source: str): # defined an function called enabled trigger
    features.TriggerSelector.value = "FrameStart" # consider TriggerSelector as FrameStart
    features.TriggerMode.value = "On" # consider TriggerMode as On
    features.TriggerSource.value = "Software" # consider TriggerSource as Line1


def enable_software_trigger(features: NodeMap):
    enable_trigger(features, "Software") # enable software trigger


def enable_hardware_trigger(features: NodeMap):
    enable_trigger(features, "Line1") # enable hardware trigger
