import os
import csv
import yaml
import shutil
import torch
import argparse
from datetime import datetime
from ultralytics import YOLO
from preprocess import get_positive_number
from pipeline_functions.utils import *
from pipeline_functions.annotation_window import annotation_window_wrapper

'''
File containing functions to train an object detection model in an active learning loop.
'''

# Helper functions

def get_base_model_name():
    base_model_name = "yolo11s.pt"
    print("Would you like to use the default base model for object detection (yolo11s)?")
    print(">>> Press Enter to use default, or enter 'n' to choose another model.")
    answer = input()
    if answer=='n':
        print("")
        print("Which size YOLO model would you like to use:")
        print("n: yolo11n model, 2.6M params, 6.5B FLOPs")
        print("s: yolo11s model, 9.4M params, 21.5B FLOPs")
        print("m: yolo11m model, 20.1M params, 68.0B FLOPs")
        print("l: yolo11l model, 25.3M params, 86.9B FLOPs")
        print("x: yolo11x model, 56.9M params, 194.9B FLOPs")
        print("")
        print(">>> Please enter a valid letter, e.g. 's'")
        options=["n","s","m","l","x"]
        while True:
            answer = str(input())
            if not answer in options:
                print("Invalid answer. Please try again.")
            else:
                base_model_name = "yolo11"+answer+".pt"
                break
    return base_model_name


def append_row_to_csv(csv_path, eval_split, iteration, num_train_images, map50_95, map50, map75):
    """
    Append a row of data to the CSV file.
    """
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([eval_split, iteration, num_train_images, map50_95, map50, map75])


def change_conf_thresh(automatic_accept_thresh, human_review_thresh):
    print("Do you want to adjust confidence thresholds for addition to the Train set?")
    print(">>> Please press Enter to continue without adjusting, or 'y' to adjust the thresholds")
    answer = input()
    if answer=='y':
        print(">>> Please enter an automatic acceptance threshold: above this, machine labels will automatically be added to the Train set.")
        print("    Values must be between 0 and 1, [default: 0.85]")
        while True:
            answer = float(input())
            if answer<=0 or answer>=1:
                print("Invalid value entered. Please try again.")
            else:
                automatic_accept_thresh = answer
                break
        print(">>> Please enter an human review threshold: above this, any labels not automatically accepted will be reviewed by you, the user, and added to the Train set.")
        print("    Values must be between 0 and 1, and less than the automatic acceptance threshold, [default: 0.60]")
        while True:
            answer = float(input())
            if answer<=0 or answer>=automatic_accept_thresh:
                print("Invalid value entered. Please try again.")
            else:
                human_review_thresh = answer
                break
    return automatic_accept_thresh, human_review_thresh


def train_set_updater(automatic_accept_thresh, human_review_thresh, run_iter_path):
    ''' Manages machine inferences by conf level and adds to train set or passes to human review'''
    line = "<>"*20

    # Handle empty folder
    labels_path = "data/temp/labels"
    label_files_list = os.listdir(labels_path)
    if label_files_list==[]:
        print(line)
        print("No detections found across all images in the unlabelled pool!")
        print("Terminating active learning loop.")
        print(line,'\n')
        # extension: give user more options
        return 1

    # Ask if user wants to change conf thresholds, change
    automatic_accept_thresh, human_review_thresh = change_conf_thresh(automatic_accept_thresh, human_review_thresh)

    # Final destinations
    labels_dest_path = os.path.join(run_iter_path,"labels/train")
    images_dest_path = os.path.join(run_iter_path,"images/train")
    os.makedirs(labels_dest_path, exist_ok=True)
    os.makedirs(images_dest_path, exist_ok=True)

    automatic_accept_count = 0
    human_review_list = []
    # Iterate through all labels and process by confidence level
    for label_file_name in label_files_list:
        label_file_path = os.path.join(labels_path, label_file_name)
        with open(label_file_path, "r") as f:
            for line in f:
                # Read label txt in YOLO format: class_id x_center y_center width height conf
                _, _, _, _, _, conf = map(float, line.strip().split())
        # Handle high confidence detections (Auto-accept)
        if conf > automatic_accept_thresh:
            automatic_accept_count += 1
            # Move label to train set
            shutil.move(src=label_file_path, dst=labels_dest_path)
            # Get right source image path
            image_name = os.path.splitext(os.path.basename(label_file_name))[0]+'.jpg'
            image_path = os.path.join(run_iter_path, "images/unlabelled",image_name)
            # Move image
            shutil.move(src=image_path,dst=images_dest_path)
        # Deal with human review images
        elif conf > human_review_thresh:
            human_review_list.append(label_file_name)

    if automatic_accept_count>0:
        print(f"Moved {automatic_accept_count} annotated images that passed the confidence threshold into the Train set.")
        print("")

    if human_review_list:
        print("")
        print(f"There are {len(human_review_list)} many images flagged for review. How many do you want to review?")
        print(">>> Please enter a positive number." )
        num_to_review = get_positive_number(max_val=len(human_review_list)+1)
        print("")
        # Trim list
        human_review_list = human_review_list[:num_to_review]
        
        # Cleanup unused labels
        for label_file_name in label_files_list:
            file_to_remove = os.path.join(labels_path, label_file_name)
            if label_file_name in human_review_list:
                #print(f"Preserving file: {label_file_name}")
                pass
            elif os.path.exists(file_to_remove):
                os.remove(file_to_remove)
            else:
                pass
                #print(f"Warning: File {file_to_remove} not found for deletion.")

        # Prepare review images folder
        review_images_dir = "data/temp/review_images"
        if os.path.exists(review_images_dir):
            for file_name in os.listdir(review_images_dir):
                file_path = os.path.join(review_images_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error while deleting file {file_path}: {e}")
        else:
            os.makedirs(review_images_dir, exist_ok=True)

        # Move images for human review
        human_review_image_path_list = []
        for label_file_name in human_review_list:
            image_name = os.path.splitext(label_file_name)[0] + '.jpg'
            image_path = os.path.join(run_iter_path, "images/unlabelled", image_name)
            # Move image into review images folder
            shutil.move(src=image_path, dst=review_images_dir)
            new_path = os.path.join(review_images_dir, image_name)
            human_review_image_path_list.append(new_path)

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
        print("Press any key to start annotation")
        answer = input()

        # Now make the human review the annotations
        annotation_window_wrapper(images_folder=review_images_dir, labels_folder=labels_path)

        # After human review, move contents of each list into their respective destinations
        for label_file_path in human_review_list:
            src_path = os.path.join(labels_path, label_file_path)
            if os.path.exists(src_path):
                shutil.move(src=src_path, dst=labels_dest_path)

        for image_file_path in human_review_image_path_list:
            if os.path.exists(image_file_path):
                shutil.move(src=image_file_path, dst=images_dest_path)

    return 0
















# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
















# Main function

def active_learning_loop(args):
    # Unpack user args 
    total_loop_iterations = args.num_loop_iters # default 3
    num_epochs = args.epochs_per_iter # default 5
    show_in_window = args.show_in_window
    init_learning_rate = args.init_learning_rate
    freeze_up_to = args.freeze_up_to
    batch_size = args.batch_size
    num_workers = args.num_workers
    automatic_accept_thresh = args.automatic_accept_thresh
    human_review_thresh = args.human_review_thresh
    device = args.device # default 'cpu'

    line = '<>'*20
    print('')
    print(line)
    print("Object detection model training")
    print(line, '\n')
    print("")

    # Get computation device
    device = select_device(device=device)
    print("(Press Ctrl+C at any time to terminate this program.)")

    # Get dataset path
    print("Are you using a custom dataset or the preprocessed dataset?")
    print(">>> Press Enter for preprocessed, or enter 'c' for custom")
    dataset="preprocessed"
    answer = str(input())
    if answer=='c':
        dataset="custom"
    print(f"Using {dataset} dataset.")
    dataset_parent_dir = os.path.join("data",dataset,"frames")

    # Setup paths and folders for runs, models depending on existing runs
    # See pipeline_functions/utils.py for function, lots going on.
    run_datetime, current_iter, run_path, models_path, run_iter_path, models_iter_path = get_run_models_paths(dataset_parent_dir)

    # Get base model
    base_model_name = get_base_model_name()
    base_model_path = os.path.join("models/base", base_model_name)
    classes=[32]

    # Initialise custom CSV for recording results
    csv_path = os.path.join(run_path, "metrics_log.csv")
    if not os.path.exists(csv_path):  # Only create the file if it doesn't exist
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["eval_split", "iteration", "num_train_images", "map50_95", "map50", "map75"])  # Header row

    # Main active learning loop
    print("")
    print("Commencing active learning loop:")
    print("")

    loop_iterations_completed = 0
    for loop_iteration_num in range(total_loop_iterations):

        print(line)
        print(f"Iteration {current_iter}/{total_loop_iterations-1}.")
        print(line, '\n')
        
        '''
        Model training on current set of annotated Train data
        '''
        # Reset to base model before training
        model = YOLO(base_model_path)
        model.to(device)
        # Setup yaml file for YOLO
        yaml_name = 'dataset.yaml'
        yaml_path = os.path.join(run_iter_path, yaml_name)
        create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='val')
        
        print("Refining base model by training on current Train set, then evaluating on Validation set:")
        print("")
        
        # Train model
        results = model.train(data=yaml_path, 
                              project=models_iter_path, name="training", exist_ok=True,
                              classes=classes, max_det=1,
                              optimizer="AdamW",
                              lr0=init_learning_rate, freeze=freeze_up_to,
                              resume=False, verbose=False,
                              epochs=num_epochs, batch=batch_size, workers=num_workers)
        
        # Write to CSV
        num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
        append_row_to_csv(csv_path, 'val', current_iter, num_train_images, results.box.map, results.box.map50, results.box.map75)

        # Print 
        print("")
        print("Model finished training with mean average precision:")
        print(results.box.map) # mean average precision 50-95 conf
        print("")
        
        print(line)
        print("Inferring on unlabelled pool")
        print(line,'\n')
        unlabelled_pool_path = os.path.join(run_iter_path, "images/unlabelled")
        num_unlabelled = len(os.listdir(unlabelled_pool_path))

        print("We now use the refined model we've just trained to predict annotations on our unseen unlabelled pool set.")
        print(f"This will take a couple of minutes ({num_unlabelled} frames). Please wait...")
        print("(Ignore following warning message)")
        print("")

        # Empty destination folder: data/temp/labels
        labels_path = "data/temp/labels"
        if os.path.exists(labels_path):
            for file_name in os.listdir(labels_path):
                file_path = os.path.join(labels_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error while deleting file {file_path}: {e}")
        else:
            os.makedirs(labels_path, exist_ok=True)

        # Infer on unlabelled pool - store inference labels in data/temp/labels
        results = model.predict(source=unlabelled_pool_path,
                                show=show_in_window,
                                max_det=1,
                                classes=classes,
                                save_txt=True, save_conf=True,
                                verbose=False,
                                project="data", name="temp",
                                exist_ok = True)
        print("")
        print(line)
        print("Refined model inference on unlabelled set finished.")
        print(line)
        print("")
        print("We now go through the model predictions and add them to the Train set depending on their confidence.")


        # Move images and copy labels into new iteration folder ready for next round of annotation
        current_iter, run_iter_path, models_iter_path = setup_next_iteration_folder(current_iter, run_path, models_path, run_iter_path, models_iter_path)

        # Move automatic accepted images, get human to annotate rest
        flag = train_set_updater(automatic_accept_thresh, human_review_thresh, run_iter_path)
        if flag==1:
            break
        
        # Ask to continue loop
        loop_iterations_completed += 1
        print("")
        print("Do you want to continue to the next iteration?")
        print(">>> Press Enter to continue, or enter 's' to skip to final training.")
        answer = input()
        if answer=='s':
            break
        
        # Break conditions
        # If unlabelled pool empty, break
        unlabelled_pool_path = os.path.join(run_iter_path, 'images/unlabelled')
        unlabelled_pool_contents = os.listdir(unlabelled_pool_path)
        if unlabelled_pool_contents==[]:
            print(line)
            print("Unlabelled pool is empty! No more images to incorporate into the Train set.")
            print("Terminating loop.")
            print(line,'\n')
            break
    
    '''
    Final Model training and evaluation
    '''

    print(">>> Please press Enter to continue to training a final model on the final Train set.")
    answer = input()

    print("")
    print(line)
    print(f"Final model training")
    print(line, '\n')

    print("Refining base model by training on final Train set, then evaluating on Validation set:")
    print("")

    # Train on final Train set now:
    # Reset to base model before training
    model = YOLO(base_model_path)
    model.to(device)
    # Setup yaml file for YOLO,
    yaml_path = os.path.join(run_iter_path, yaml_name)
    create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='val')
    
    # Train model
    results = model.train(data=yaml_path, 
                            project=models_iter_path, name="training", exist_ok=True,
                            classes=classes,
                            optimizer="AdamW",
                            lr0=init_learning_rate, freeze=freeze_up_to,
                            resume=False, verbose=False,
                            epochs=num_epochs, batch=batch_size, workers=num_workers)
    
    # Write to CSV
    num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
    append_row_to_csv(csv_path, 'val', current_iter, num_train_images, results.box.map, results.box.map50, results.box.map75)
    
    print("")
    print("Final model finished training with mean average precision on Validation set:")
    print(results.box.map) # mean average precision 50-95 conf

    # Plot validation map50-95 scores as function of iteration and train set size, save as jpg in model/iterationX

    print("")
    print("Now evaluating model on our Test set:")
    print("")

    # Setup yaml file for YOLO, this time using Test set as the evaluation split
    create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='test')
    # Get validation metrics
    metrics = model.val(data=yaml_path, 
                            project=models_iter_path, name="evaluation_test_set", exist_ok=True,
                            classes=classes, max_det=1,
                            save_json=True, verbose=False,
                            batch=batch_size)
    # Write to CSV
    num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
    append_row_to_csv(csv_path, 'test', current_iter, num_train_images, metrics.box.map, metrics.box.map50, metrics.box.map75)

    print("")
    print("Final model has a mean average precision on the Test set of:")
    print(metrics.box.map) # mean average precision 50-95 conf

    final_model_path = os.path.join(models_iter_path,"training/weights/best.pt")
    print("")
    print(line)
    print("Final model available at:")
    print(final_model_path)
    print(line)
    print("")
    
    # Ask user if want to use final model on sample video from Test set
    print("Do you want to apply the final model to a video from the Test set?")
    print(">>> Please press Enter to see a list of available videos, or enter 'n' to finish.")
    answer = input()
    if answer=='n':
        return
    print("Please select a video:")
    test_set_videos_dir = os.path.join("data",dataset,"videos/test")
    vid_dict = {}
    for idx, vid_name in enumerate(os.listdir(test_set_videos_dir)):
        vid_dict[idx+1] = vid_name
        # Get number of frames
        vid_path = os.path.join(test_set_videos_dir, vid_name)
        length = get_num_vid_frames(vidpath=vid_path)
        if not length is None:
            print(f"{idx+1}) {vid_name}, ({length} frames)")
        else:
            print(f"{idx+1}) {vid_name}")
    print(">>> Please enter a video number.")
    vid_num = get_positive_number(max_val=idx+2)
    vid_path = os.path.join(test_set_videos_dir, vid_dict[vid_num])

    # Apply final model to video
    results = model.predict(source=vid_path, vid_stride=1,
                        classes=classes,
                        save=True, save_txt=False, save_conf=False,
                        project=models_iter_path, name='final_model_annotated_video')
    
    print("Do you want to also apply the base model to the same video, for comparison?")
    print(">>> Please press Enter to apply the base model to the same video, or enter 'n' to finish.")
    answer = input()
    if answer=='n':
        return
    
    # Apply base model to same video
    results = model.predict(source=vid_path, vid_stride=1,
                        classes=classes,
                        save=True, save_txt=False, save_conf=False,
                        project=models_iter_path, name='base_model_annotated_video')
    
    print("Done!")
    print("")
    print(line)
    print("Final model available at:")
    print(final_model_path)
    print(line)
    print("")




if __name__=="__main__":
    # Parse input arguments
    line="<>"*20
    description = line +'\n'+"train.py input arguments " + line
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--show_in_window', type=bool, required=False, default=False, help="Whether to show newly annotated frames/videos as they are predicted in a seperate window. (Default: True)")
    parser.add_argument('--num_loop_iters', type=int, required=False, default=3, help="The number of active learning loop iterations. This is the number of times you will have to review machine annotations. (Default: 3)")
    parser.add_argument('--epochs_per_iter', type=int, required=False, default=1, help="The number of model training epochs performed in each iteration of the loop. (Default: 5)")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="The batch size used during model training in each iteration of the loop (Default: 16)")
    parser.add_argument('--init_learning_rate', type=float, required=False, default=10e-5, help="The initial learning rate used during model training in each iteration of the loop (Default: 10e-5)")
    parser.add_argument('--num_workers', type=int, required=False, default=8, help="The number of worker threads for dataloading during model training in each iteration of the loop (Default: 8)")
    parser.add_argument('--freeze_up_to', type=int, required=False, default=None, help="Optional number of layers in the YOLOv11 model to freeze during training in each iteration of the loop. Enter 9 to freeze model backbone, and 22 to freeze everything but the object detection head. (Default: None)")
    parser.add_argument('--automatic_accept_thresh', type=float, required=False, default=0.80, help="Predictions with confidence scores above this threshold are automatically accepted into the training set as pseudo-labels.(Default: 0.80)")
    parser.add_argument('--human_review_thresh', type=float, required=False, default=0.4, help="Predictions with scores below the acceptance threshold but above a lower threshold are flagged for manual review before inclusion in the training set. (Default: 0.40)")
    parser.add_argument('--device', type=str, required=False, default='cpu', const='cpu', nargs='?', choices=['cpu', 'cuda', 'mps'], help="Device to perform model computations on, either 'cpu' for CPU, 'cuda' for GPU with CUDA compatibility, or 'mps' for Apple MPS on a Mac Metal GPU. (Default: 'cpu')")
    args = parser.parse_args()
    active_learning_loop(args)

