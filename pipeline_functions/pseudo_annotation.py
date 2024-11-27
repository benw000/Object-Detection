import os
import shutil
import subprocess
from ultralytics import YOLO

'''
File containing pipeline to create pseudo-annotations from set of downloaded video frames.
'''

# Helper functions












# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
















# Main pseudo annotation pipeline function

def pseudo_annotation_manager():
    pass

    # At this point we have 
    # data/custom/frames/[images/[train, val, test]  full of images
    # 



    # Need to pass all images through good handpicked version of YOLO
    # Ask user if any hyperparams need choosing, otherwise just do it

    # Print explainer to user, say using CVAT
    # Tell them can use with email, username and password only
    # Tell them I'll request these in script, and won't use them otherwise
    # Don't have a multi-user setup so python script just creates new personal task
    # Python google secure user input without storing

    # Then need to put into CVAT format, create new project, upload as task, with guide instructions 
    # in the case of no detections

    # Ask when done then download, reformat into labels folder .txt files 
    # Create yaml file for YOLO