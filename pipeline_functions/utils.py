import os
import yaml
import torch
import shutil
import datetime

def create_yaml(yaml_path, parent_folder, train_subset_name, valid_subset_name):
    """
    Create yaml file needed to specify dataset to YOLO models.
    """
    yaml_content = {
        "path": os.path.abspath(parent_folder),
        "train": f"images/{train_subset_name}",
        "val": f"images/{valid_subset_name}",
        "names": {
            32: "sports ball"
        }
    }
    # sports ball 32 could be wrong, could just be 0?

    # Overwrite existing file
    with open(yaml_path, "w") as file:
        yaml.dump(yaml_content, file, default_flow_style=False)

    '''
    yaml_path="data/custom/current_dataset.yaml
    # Create yaml dataset config file for YOLO to read
        create_yaml(
            yaml_path=yaml_path,
            parent_folder=images_parent_dir,
            train_subset_name=subset_name,
            valid_subset_name=val_name
        )
    '''

def select_device(device):
    """
    Prompt the user to select a device (GPU, Apple MPS, or CPU), 
    check availability, and return the corresponding torch device.
    """
   
    # Seems like we need this line:
    torch.cuda.set_device(0)

    # Check and assign the device
    if device == 'cuda':
        if torch.cuda.is_available():
            print("Using GPU (CUDA).")
            return torch.device("cuda")
        else:
            print("CUDA is not available. Defaulting to CPU.")
            return torch.device("cpu")
    elif device == 'mps':
        if torch.backends.mps.is_available():
            print("Using Apple MPS.")
            return torch.device("mps")
        else:
            print("Apple MPS is not available. Defaulting to CPU.")
            return torch.device("cpu")
    else:
        print("Using CPU.")
        return torch.device("cpu")

'''
# Get GPU/CPU/MPS device from user
    select_device()
    # Set up YOLO model
    base_model_path = os.path.join("models","base","yolo11n.pt")
    model = YOLO(base_model_path)
    model.to(device=device)
'''


def get_run_model_paths(dataset_parent_dir):
    ''' Gets an existing run datetime (eg) 30-11_13_45, or records current time.
        Then sets up various paths and folders'''
    
    # Get run datetime
    now = datetime.now()
    run_datetime = now.strftime("%d_%m-%H_%M")

    # Ask user to pick an existing run if they exist
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
    
    # Get path to run and model run and ensure exists
    run_path = os.path.join("runs", run_datetime) # runs/{datetime}
    model_path = os.path.join("models", run_datetime) # models/{datetime}
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

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
    model_iter_path = os.path.join(model_path, iter_name) # models/{datetime}/iterationX
    os.makedirs(run_iter_path, exist_ok=True)
    os.makedirs(model_iter_path, exist_ok=True)

    # If model_iter_path not empty then empty it.
    model_iter_path_contents = os.listdir(model_iter_path)
    if not model_iter_path_contents==[]:
        for item in model_iter_path_contents:
            item_path = os.path.join(model_iter_path, item)
            shutil.rmtree(item_path)

    # If run_iter_path empty then copy dataset into it
    run_iter_path_contents = os.listdir(run_iter_path)
    if not run_iter_path_contents==[]:
        src = dataset_parent_dir # data/preprocess/frames
        dest = run_path # runs/{datetime}
        # Move source folder into destination's parent folder
        print("")
        print(f"Copying our dataset into {run_iter_path}...")
        print("")
        moved_folder_path = shutil.copy(src, dest) # moved 'frames' whole into runs/{datetime}/frames
        # Rename as destination folder (theseus' jpeg folder lol)
        os.rename(moved_folder_path, run_iter_path) # rename to runs/{datetime}/iterationX

    return run_datetime, current_iter, run_path, model_path, run_iter_path, model_iter_path