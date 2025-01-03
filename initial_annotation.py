import os
import shutil
from ultralytics import YOLO
from pipeline_functions.annotation_window import annotation_window_wrapper

# Main pseudo annotation pipeline function

def pseudo_annotation_manager():
    '''
    Main function to handle pseudo annotation in the preprocessing stage.
    Passes frames from the Train, Validation and Test subsets into YOLO and infers.
    Then gets human to manually review these.
    '''
    # Globals
    data_parent = "data/custom/frames"
    images_parent_dir = os.path.join(data_parent,"images")
    labels_parent_dir = os.path.join(data_parent,"labels")
    subset_dict = {'train':'Train', 'val':'Validation', 'test':'Test'}

    line = '<>'*20
    print('')
    print(line)
    print("Pseudo-annotation")
    print(line, '\n')
    print("(Press Ctrl+C at any time to terminate this program.)")
    print("We now pseudo-annotate the images in the Train, Test and Validation sets, using the 'YOLO11n' object detection model from Ultralytics.")
    print("This will first be downloaded locally (size ~6mb), and then it will be used to")
    print("identify the bounding boxes of any sports balls present in each image.")
    print("")
    print(">>> Please press Enter to proceed with machine pseudo-annotation, or type 's' to skip to human annotation.")
    answer = input()
    if answer=='s':
        print("Skipping...")
        print("")
    else:
        # Set up YOLO model
        base_model_path = os.path.join("models","base","yolo11n.pt")
        model = YOLO(base_model_path)

        # YOLO predict arguments
        class_list = [32]   # Isolate the 'sports ball' class 
        max_detections = 1  # Ensure only 1 object is detected (the ball)
        save_txt_bool = True     # Save annotations as txt file
        save_conf_bool = False   # save the confidence scores in the file
        # (YOLO saves annotations inside 'project/name/labels' so split up path to labels)
        project_folder =  "data"
        run_name_folder = "temp"
        overwrite_temp = True
        
        # Go through each subset and infer with YOLO
        subset_dict = {'train':'Train', 'val':'Validation', 'test':'Test'}
        for subset_name in subset_dict.keys():
            print("")
            print(f"Processing {subset_dict[subset_name]} subset.")
            print("")
            source_folder = os.path.join(images_parent_dir, subset_name)
            if not os.listdir(source_folder) == []:
                print(f"WARNING: {source_folder} contains existing annotations!")
                print("Do you want to overwrite these with machine annotations?")
                print(">>> Press Enter to continue, or hold Ctrl+C to terminate.")
                _ = input()
            # Pass subset through YOLO inference
            results = model.predict(source=source_folder,
                                max_det=max_detections,
                                classes=class_list,
                                save_txt=save_txt_bool, save_conf=save_conf_bool,
                                project=project_folder, name=run_name_folder,
                                exist_ok = overwrite_temp)
            
            # Move results into correct folder
            src_dir = "data/temp/labels"
            dest_dir = os.path.join(labels_parent_dir,subset_name) # data/custom/frames/labels/train
            if os.path.exists(dest_dir):
                # Remove empty destination directory
                shutil.rmtree(dest_dir)
            # Move source folder into destination's parent folder
            moved_folder_path = shutil.move(src_dir, labels_parent_dir) 
            # Rename as destination folder (theseus' jpeg folder lol)
            os.rename(moved_folder_path, dest_dir)
            print(f"Moved {src_dir} to {dest_dir}")
            
        print("Machine annotation complete!")

    line = '<>'*20
    print('')
    print(line)
    print("Manual review/correction")
    print(line, '\n')
    
    print("This script will launch a new window in which to perform annotations.")
    print("You will be presented with a series of image frames, ")
    print("some with machine-detected bounding boxes, overlayed on top of potential sports balls.")
    print("")

    print("INSTRUCTIONS:")
    print("    - Left click down and hold to begin drawing a bounding box rectangle, and release to finish drawing.")
    print("    - Press the 'u' key to undo the last annotation you've made, or remove any incorrect machine annotation.")
    print("    - Press the 'n' key to save that frame's annotation and move to the next frame.")
    print("    - Press the 'q' key at any point to save the current frame's annotations and quit the window.")
    print("Quitting will allow you to move to the next subset of images, and the model will assume that the ")
    print("machine annotations on the remaining frames are sufficient to train on.")
    print("If no ball is present in a frame, or it is unclear, then move on without any annotation.")
    print("")
    print("(N.B. For several hundred annotations this may take up to an hour.)")
    print("")

    # Loop through Train, Validation, Test
    for subset_name in subset_dict.keys():
        print(f">>> Press Enter to launch the {subset_dict[subset_name]} subset review window.")
        print("    or 's' to skip to the next subset.")
        answer = input()
        if not answer=='s':
            images_folder = os.path.join(images_parent_dir, subset_name)
            labels_folder = os.path.join(labels_parent_dir,subset_name)
            annotation_window_wrapper(images_folder=images_folder, labels_folder=labels_folder)
            print("")
            print(f"{subset_dict[subset_name]} subset review complete!")
            print("")
    print('')
    print(line)
    print("All image subsets reviewed!")
    print(line, '\n')
    print("Please now run <train.py> to begin training a model with this labelled data inside an active learning loop.")
    print("")
    print("Run:")
    print("python main.py")
    print("")

if __name__=="__main__":
    pseudo_annotation_manager()