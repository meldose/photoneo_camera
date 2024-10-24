from genicam.genapi import NodeMap # imported NodeMap module


def load_default_user_set(features: NodeMap): # define the function called load_default_user_set
    features.UserSetSelector.value = "Default" # set the value of UserSetSelector to Default
    features.UserSetLoad.execute() # execute the UserSetLoad feature
