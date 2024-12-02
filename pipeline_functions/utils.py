import os
import cv2
import yaml
import torch
import shutil
import datetime

def create_yaml(yaml_path, parent_folder, train_subset_name, valid_subset_name):
    """
    Create yaml file needed to specify dataset to YOLO models.
    """
    names = {i: "placeholder" for i in range(32)}
    # Assign the target class name to its corresponding ID
    names[32] = "sports ball"
    yaml_content = {
        "path": os.path.abspath(parent_folder),
        "train": f"images/{train_subset_name}",
        "val": f"images/{valid_subset_name}",
        "names": names
    }

    # Overwrite existing file
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_content, file, default_flow_style=False)


def select_device(device):
    """
    Prompt the user to select a device (GPU, Apple MPS, or CPU), 
    check availability, and return the corresponding torch device.
    """
   
    # Check and assign the device
    if device == 'cuda':
        if torch.cuda.is_available():
            print("Using GPU (CUDA).")
            print("")
            return torch.device("cuda")
        else:
            print("CUDA is not available. Defaulting to CPU.")
            print("")
            return torch.device("cpu")
    elif device == 'mps':
        if torch.backends.mps.is_available():
            print("Using Apple MPS.")
            print("")
            return torch.device("mps")
        else:
            print("Apple MPS is not available. Defaulting to CPU.")
            print("")
            return torch.device("cpu")
    else:
        print("Using CPU.")
        return torch.device("cpu")


def get_run_models_paths(dataset_parent_dir):
    ''' Gets an existing run datetime (eg) 30-11_13_45, or records current time.
        Then sets up various paths and folders'''
    
    # Get run datetime
    now = datetime.datetime.now()
    run_datetime = now.strftime("%d_%m-%H_%M")

    '''
    # Ask user to pick an existing run if they exist - not allowing this as it could cause problems.
    runs_list = os.listdir("runs")
    if not runs_list==[]:
        print("Would you like to start from an existing run?")
        print(">>> Press Enter to start a new run, or enter 'y' to load an existing run.")
        answer = input()
        if answer=='y':
            print("")
            print(">>> Please enter the date/time of the existing run, eg '30_11-14_45'.")
            while True:
                answer = str(input())
                if answer in runs_list:
                    # Overwrite datetime
                    run_datetime = answer
                    break
                else:
                    print("Run not in existing list. Please try again.")
    '''
    
    # Get path to run and model run and ensure exists
    run_path = os.path.join("runs", run_datetime) # runs/{datetime}
    models_path = os.path.join("models", run_datetime) # models/{datetime}
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    print("")
    print(f"Created new folders {run_path} and {models_path} to store datasets and models respectively.")
    print("")
    # Get current (highest) iteration present
    iter_folders = os.listdir(run_path)
    current_iter = 0
    if not iter_folders==[]:
        for item in iter_folders:
            # Isolate number - 'iterationXXX' string index 9 onwards
            iter_num = int(item[9:])
            if iter_num > current_iter:
                current_iter = iter_num

    # Initialise iterations folders within runs and models
    iter_name = f"iteration{current_iter}"
    run_iter_path = os.path.join(run_path, iter_name) # runs/{datetime}/iterationX
    models_iter_path = os.path.join(models_path, iter_name) # models/{datetime}/iterationX
    os.makedirs(run_iter_path, exist_ok=True)
    os.makedirs(models_iter_path, exist_ok=True)

    # If models_iter_path not empty then empty it.
    models_iter_path_contents = os.listdir(models_iter_path)
    if not models_iter_path_contents==[]:
        for item in models_iter_path_contents:
            item_path = os.path.join(models_iter_path, item)
            shutil.rmtree(item_path)

    # If run_iter_path empty then copy dataset into it
    run_iter_path_contents = os.listdir(run_iter_path)
    if run_iter_path_contents==[]:
        src = dataset_parent_dir # data/preprocess/frames
        dest = run_iter_path # runs/{datetime}/iterationX
        # Copy source folder into destination's parent folder
        print(f"Copying our dataset {src} into {dest}...")
        print("")
        moved_folder_path = shutil.copytree(src, dest, dirs_exist_ok=True) # moved 'frames' whole into runs/{datetime}/frames
        # Rename as destination folder (theseus' jpeg folder lol)
        os.rename(moved_folder_path, run_iter_path) # rename to runs/{datetime}/iterationX

    return run_datetime, current_iter, run_path, models_path, run_iter_path, models_iter_path


def setup_next_iteration_folder(current_iter, run_path, models_path, run_iter_path, models_iter_path):
# We now create the dataset used to train the next model iteration
    # We create a new empty folder, and move the current images into it, and copy the labels into it
    # Images are moved while labels copied in order to save disk memory
    next_iter = current_iter + 1
    next_iter_name = f"iteration{next_iter}"
    run_next_iter_path = os.path.join(run_path, next_iter_name) # runs/{datetime}/iterationX+1
    models_next_iter_path = os.path.join(models_path, next_iter_name) # models/{datetime}/iterationX+1
    os.makedirs(run_next_iter_path)
    os.makedirs(models_next_iter_path)

    # Move images
    # shutil.move moves whole folder inside the path directory
    src_images = os.path.join(run_iter_path, "images")  # runs/{datetime}/iterationX/images
    dest_images = run_next_iter_path # Ensure images are moved into their own subfolder
    print("")
    print(f"Moving dataset images from {src_images} to {dest_images}...")
    os.makedirs(dest_images, exist_ok=True)  # Ensure destination folder exists
    shutil.move(src_images, dest_images)  # Move images folder into destination

    # Copy labels
    # shutil.copy unpacks the folder into the specified directory
    src_labels = os.path.join(run_iter_path, "labels")  # runs/{datetime}/iterationX/labels
    dest_labels = os.path.join(run_next_iter_path, "labels")  # Ensure labels are copied into their own subfolder
    print(f"Copying dataset labels from {src_labels} to {dest_labels}...")
    print("")
    os.makedirs(dest_labels, exist_ok=True)  # Ensure destination folder exists
    shutil.copytree(src_labels, dest_labels, dirs_exist_ok=True)  # Copy entire labels folder

    # Update paths
    current_iter=next_iter
    run_iter_path, models_iter_path = run_next_iter_path, models_next_iter_path

    return current_iter, run_iter_path, models_iter_path


def get_num_vid_frames(vidpath):
    try:
        cap = cv2.VideoCapture(vidpath)
        if cap.isOpened(): 
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            length = None
    except:
        length = None