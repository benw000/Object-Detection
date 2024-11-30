import os
import yaml
import torch

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

def select_device():
    """
    Prompt the user to select a device (GPU, Apple MPS, or CPU), 
    check availability, and return the corresponding torch device.
    """
    print("Please select a device for computation:")
    print("1: GPU (if available and CUDA compatible)")
    print("2: Apple MPS (if available, for macOS users)")
    print("3: CPU (default)")

    print("")
    print(">>> Please enter '1', '2' or '3'.")
    
    while True:
        try:
            choice = int(input())
            break
        except ValueError:
            print("Invalid input.")

    # Seems like we need this line:
    torch.cuda.set_device(0)

    # Check and assign the device
    if choice == 1:
        if torch.cuda.is_available():
            print("Using GPU (CUDA).")
            return torch.device("cuda")
        else:
            print("CUDA is not available. Defaulting to CPU.")
            return torch.device("cpu")
    elif choice == 2:
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