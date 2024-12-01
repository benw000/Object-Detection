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

def append_row_to_csv(csv_path, eval_split, iteration, num_train_images, map50_95, map50, map75):
    """
    Append a row of data to the CSV file.
    """
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([eval_split, iteration, num_train_images, map50_95, map50, map75])



def change_conf_thresh():
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

    # Deal with empty folder
    labels_path = "data/temp/labels"
    label_files_list = os.listdir(labels_path)
    if label_files_list==[]:
        print("No detections found across all images in the unlabelled pool!")
        print("Terminating active learning loop.")
        # extension: give user more options
        return 1

    # Final destinations
    labels_dest_path = os.path.join(run_iter_path,"labels/train")
    images_dest_path = os.path.join(run_iter_path,"images/train")

    # Iterate through all labels and process by confidence level
    automatic_accept_count = 0
    human_review_count = 0
    human_review_list = []
    for label_file_name in label_files_list:
        label_file_path = os.path.join(labels_path, label_file_name)
        with open(label_file_path, "r") as f:
            for line in f:
                # Read label txt in YOLO format: class_id x_center y_center width height conf
                _, _, _, _, _, conf = map(float, line.strip().split())
        # Deal with automatic accepted images
        if conf > automatic_accept_thresh:
            automatic_accept_count += 1
            # Move label to train set
            shutil.move(src=label_file_path,dst=labels_dest_path)
            # Get right source image path
            image_name = os.path.splitext(os.path.basename(label_file_name))[0]+'.jpg'
            image_path = os.path.join(run_iter_path, "images/unlabelled",image_name)
            # Move image
            shutil.move(src=image_path,dst=images_dest_path)
        # Deal with human review images
        elif conf > human_review_thresh:
            human_review_count += 1
            human_review_list.append(label_file_name)

    print("")
    print(f"There are {len(human_review_list)} many images flagged for review. How many do you want to review?")
    print(">>> Please enter a positive number." )
    num_to_review = get_positive_number(max_val=len(human_review_list)+1)
    print("")
    # Trim list
    human_review_list = human_review_list[:num_to_review]

    # Remove labels files in data/temp/labels that aren't on the trimmed list
    human_review_label_path_list = []
    for label_file_name in label_files_list:
        # Get path 
        label_file_path = os.path.join(labels_path, label_file_name)
        # If label is valid, add to list
        if label_file_name in human_review_list:
            human_review_label_path_list.append(label_file_path)
        # Else delete label
        else:
            label_file_path = os.path.join(labels_path, label_file_name)
            os.remove(label_file_path)

    # Make new review images folder
    review_images_dir = "data/temp/review_images"
    if os.path.exists(review_images_dir):
        shutil.rmtree(review_images_dir)
    os.makedirs(review_images_dir)

    # For each label file put its corresponding image in the review_images folder
    human_review_image_path_list = []
    for label_file_name in human_review_list:
        # Get right source image path
        image_name = os.path.splitext(os.path.basename(label_file_name))[0]+'.jpg'
        image_path = os.path.join(run_iter_path, "images/unlabelled",image_name)
        # Move image into review images folder
        shutil.move(src=image_path,dst=review_images_dir)
        # Make note of its path
        new_path = os.path.join(review_images_dir,image_name)
        human_review_image_path_list.append(new_path)

    # Now make the human review the annotations
    annotation_window_wrapper(images_folder=review_images_dir, labels_folder=labels_path)

    # After this is done, move contents of each list into destination
    for label_file_path in human_review_label_path_list:
        shutil.move(src=label_file_path, dst=labels_dest_path)

    for image_file_path in human_review_image_path_list:
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
    dataset = args.dataset # default 'preprocessed'
    batch_size = args.batchsize
    num_workers = args.num_workers
    automatic_accept_thresh = args.automatic_accept_thresh
    human_review_thresh = args.human_review_thresh
    #device = args.device # default 'cpu'

    line = '<>'*20
    print('')
    print(line)
    print("Object detection model training")
    print(line, '\n')
    print("(Press Ctrl+C at any time to terminate this program.)")
    print("")

    # Get dataset path
    dataset_parent_dir = os.path.join("data",dataset,"frames")

    # Setup paths and folders for runs, models depending on existing runs
    # See pipeline_functions/utils.py for function, lots going on.
    run_datetime, current_iter, run_path, models_path, run_iter_path, models_iter_path = get_run_models_paths(dataset_parent_dir)

    # Setup base model
    base_model_name = "yolo11n.pt"
    base_model_path = os.path.join("models/base", base_model_name)

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
        print(f"Iteration {current_iter+1}/{current_iter+total_loop_iterations}.")
        print(line, '\n')

        # Break conditions
        # If unlabelled pool empty, break
        unlabelled_pool_path = os.path.join(run_iter_path, 'unlabelled')
        unlabelled_pool_contents = os.listdir(unlabelled_pool_path)
        if unlabelled_pool_contents==[]:
            print(line)
            print("Unlabelled pool is empty! No more images to incorporate into the Train set.")
            print("Terminating loop.")
            print(line,'\n')
            break
        
        '''
        Model training on last set of annotated Train data
        '''
        # Setup yaml file for YOLO
        yaml_name = 'dataset.yaml'
        yaml_path = os.path.join(run_iter_path, yaml_name)
        create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='val')
        # Project name
        project = models_iter_path
        name = "training" # training outputs stored in project/name
        exist_ok = True  # ok to overwrite existing training logs
        classes=[32]
        resume=False
        # Reset to base model before training
        model = YOLO(base_model_path)

        # Train model
        results = model.train(data=yaml_path, 
                              project=project, name="training", exist_ok=exist_ok,
                              classes=classes,
                              resume=resume,
                              epochs=num_epochs, batch=batch_size, workers=num_workers)
        
        # Get validation metrics
        metrics = model.val(data=yaml_path, 
                            project=project, name="evaluation", exist_ok=exist_ok,
                            classes=classes, max_det=1,
                            save_json=True,
                            batch=batch_size,)
        # Write to CSV
        num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
        append_row_to_csv(csv_path, 'val', current_iter, num_train_images, metrics.box.map, metrics.box.map50, metrics.box.map75)

        # Print 
        print("")
        print("Model finished training with mean average precision:")
        print(metrics.box.map) # mean average precision 50-95 conf
        print("")

        # Move images and copy labels into new iteration folder ready for next round of annotation
        current_iter, run_path, models_path, run_iter_path, models_iter_path = setup_next_iteration_folder(current_iter, run_path, models_path, run_iter_path, models_iter_path)
        
        print(line)
        print("Inferring with model on unlabelled pool")
        print(line,'\n')
        # Infer on unlabelled pool - store inference labels in data/temp/labels
        src = os.path.join(run_iter_path, "images/unlabelled")
        project = "data"
        name = "temp"
        results = model.predict(source=src,
                                max_det=1,
                                classes=classes,
                                save_txt=True, save_conf=True,
                                project=project, name=name,
                                exist_ok = True)
        print("")
        print("Inference on unlabelled set finished.")
        print("")
        
        # Ask if user wants to change conf thresholds, change
        automatic_accept_thresh, human_review_thresh = change_conf_thresh()

        # Move automatic accepted images, get human to annotate rest
        flag = train_set_updater(automatic_accept_thresh, human_review_thresh, run_iter_path)
        if flag==1:
            break
        
        # Ask to continue loop
        loop_iterations_completed += 1
        print("Do you want to continue?")
        print(">>> Press Enter to continue, or enter 's' to skip to final training.")
        answer = input()
        if answer=='s':
            break

    # TODO: Print message here about done and what next
    # Check this by actually running for a bunch of different cases
    # Split the end into evaluate.py for the video bit etc maybe?
    # Make file that takes a video mp4 path and chucks out rendered one with best model given
    # Find out where best model is stored, print this programmatically

    # Active loop finished. Train on final Train set now:
    # Setup yaml file for YOLO,
    yaml_path = os.path.join(run_iter_path, yaml_name)
    create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='val')
    # Project name
    project = models_iter_path
    name = "training" # training outputs stored in project/name
    exist_ok = True  # ok to overwrite existing training logs
    classes=[32]
    resume=False
    # Reset to base model before training
    model = YOLO(base_model_path)
    # Train model
    results = model.train(data=yaml_path, 
                            project=project, name="training", exist_ok=exist_ok,
                            classes=classes,
                            resume=resume,
                            epochs=num_epochs, batch=batch_size, workers=num_workers)
    # Get validation metrics
    metrics = model.val(data=yaml_path, 
                            project=project, name="evaluation", exist_ok=exist_ok,
                            classes=classes, max_det=1,
                            save_json=True,
                            batch=batch_size,)
    # Write to CSV
    num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
    append_row_to_csv(csv_path, 'val', current_iter, num_train_images, metrics.box.map, metrics.box.map50, metrics.box.map75)


    # Plot validation map50-95 scores as function of iteration and train set size, save as jpg in model/iterationX


    # Setup yaml file for YOLO, this time using Test set as the evaluation split
    create_yaml(yaml_path, parent_folder=run_iter_path, train_subset_name='train', valid_subset_name='test')
    # Get validation metrics
    metrics = model.val(data=yaml_path, 
                            project=project, name="evaluation_test_set", exist_ok=exist_ok,
                            classes=classes, max_det=1,
                            save_json=True,
                            batch=batch_size,)
    # Write to CSV
    num_train_images = len(os.listdir(os.path.join(run_iter_path,"images/train")))
    append_row_to_csv(csv_path, 'test', current_iter, num_train_images, metrics.box.map, metrics.box.map50, metrics.box.map75)
    
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
        vid_dict[idx] = vid_name
        print(f"{idx}) {vid_name}")
    print(">>> Please enter a video number.")
    vid_num = get_positive_number(max_val=idx+1)
    vid_path = os.path.join(test_set_videos_dir, vid_dict[vid_num])

    # Apply final model to video
    results = model.predict(source=vid_path, vid_stride=1,
                        classes=[32],
                        save=True, save_txt=False, save_conf=False,
                        project=project, name='final_model_annotated_video')
    
    print("Do you want to also apply the base model to the same video, for comparison?")
    print(">>> Please press Enter to apply the base model to the same video, or enter 'n' to finish.")
    answer = input()
    if answer=='n':
        return
    
    # Apply base model to same video
    results = model.predict(source=vid_path, vid_stride=1,
                        classes=[32],
                        save=True, save_txt=False, save_conf=False,
                        project=project, name='base_model_annotated_video')
    
    print("Done!")




if __name__=="__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_loop_iters', type=int, required=False, default=3, help="The number of active learning loop iterations. This is the number of times you will have to review machine annotations. (Default: 3)")
    parser.add_argument('--epochs_per_iter', type=int, required=False, default=5, help="The number of model training epochs performed in each iteration of the loop. (Default: 5)")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="The batch size used during model training in each iteration of the loop (Default: 16)")
    parser.add_argument('--num_workers', type=int, required=False, default=8, help="The number of worker threads for dataloading during model training in each iteration of the loop (Default: 8)")
    parser.add_argument('--automatic_accept_thresh', type=float, required=False, default=0.85, help="Predictions with confidence scores above this threshold are automatically accepted into the training set as pseudo-labels.(Default: 0.85)")
    parser.add_argument('--human_review_thresh', type=float, required=False, default=0.6, help="Predictions with scores below the acceptance threshold but above a lower threshold are flagged for manual review before inclusion in the training set. (Default: 0.60)")
    parser.add_argument('--dataset', type=str, required=False, default='preprocessed', const='preprocessed', nargs='?', choices=['preprocessed', 'custom'], help="The dataset you are using, either 'preprocessed' or 'custom', depending on if you created your own dataset. (Default: preprocessed)")
    #parser.add_argument('--device', type=str, required=False, default='cpu', const='cpu', nargs='?', choices=['cpu', 'cuda', 'mps'], help="Device to perform model computations on, either 'cpu' for CPU, 'cuda' for GPU with CUDA compatibility, or 'mps' for Apple MPS on a Mac Metal GPU. (Default: 'cpu')")
    args = parser.parse_args()
    active_learning_loop(args)




    # TODO: Print message after main loop about done and what next
    # Check this by actually running for a bunch of different cases
    # Split the end into evaluate.py for the video bit etc maybe?
    # Make seperate file that takes a video mp4 path and chucks out rendered one with best model given
    # Find out where best model is stored, print this programmatically
    # Decide which function go in utils.py if any. Can more go in to justify having the setup?
    # Make plot function that reads the csv and plots
    # Explain meaning of mAP
    # Explain process and some of the functions better
    # Get screen record gifs of annotation process
    # Run a few times with low iterations with different UX paths, make sure robust
    # If excess time, try on windows PC upstairs
    # Write up email to them and explain rationale, drawbacks etc.