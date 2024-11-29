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
    # Set up YOLO model, will download to base_model_path if not present
    base_model_path = os.path.join("models","base","yolo11n.pt")
    model = YOLO(base_model_path)

    ''' YOLO predict arguments: '''
    user_device = "cpu" # Pass this as optional , use 'cuda' if they have GPU
    class_list = [32]   # Isolate the 'sports ball' class 
    max_detections = 1  # Ensure only 1 object is detected (the ball)
    save_txt_bool = True     # Save annotations as txt file
    save_conf_bool = True   # save the confidence scores in the file
    # Source folder of images
    images_folder =  'z_delete' #"data/preprocessed/frames/images/test"
    # YOLO saves annotations inside 'project/name/labels' so split up path to labels
    project_folder =  'z_delete_labels'#"data/preprocessed/frames/labels"
    run_name_folder = "test"

    # Perform a YOLO prediction run
    results = model.predict(source=images_folder,
                            device=user_device,
                            max_det=max_detections,
                            classes=class_list,
                            save_txt=save_txt_bool, save_conf=save_conf_bool,
                            project=project_folder, name=run_name_folder)

    # Need to pass custom/frames/[train,val,test] through YOLO
    # Call formatting_utils to put into YOLO format with yaml file.
    # then use YOLO model, good handpicked version, to infer on all

    # Next print CVAT explainer:
    # Tell them can use with email, username and password only
    # Tell them I'll request these in script, and won't use them otherwise
    # Don't have a multi-user setup so python script just creates new personal task
    # Python google secure user input without storing

    # Then need to put into CVAT format, create new project, upload as task, 
    # with guide instructions in the case of no detections.

    # Ask when done then download, reformat into YOLO format with formatting_utils function.

pseudo_annotation_manager()