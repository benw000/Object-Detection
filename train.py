import os
import yaml
import shutil
import torch
import argparse
from datetime import datetime
from ultralytics import YOLO
from pipeline_functions.utils import *
from pipeline_functions.annotation_window import annotation_window_wrapper

'''
File containing functions to train an object detection model in an active learning loop.
'''

# Helper functions

def train_model(yolo_model_object):
    ''' General YOLO training wrapper. '''
    # Package dataset by creating YAML file

    create_yaml(yaml_path, parent_folder, train_subset_name, valid_subset_name)

    # Train
    results = yolo_model_object.train(data="coco8.yaml", epochs=2, imgsz=640, device='cpu', workers=0)
    # then evaluate and store this
    #validation_metrics = yolo_model_object.val(data="coco8.yaml", max_det=1, device=device, project, name)
    print(validation_metrics.box.map) # mean average precision 50-95 conf


def model_inference():
    ''' Infers on unlabelled pool, updates folders'''

def train_set_updater():
    ''' Manages machine inferences by conf level and adds to train set or passes to human review'''

def model_evaluation(model_path: str, dataset_yaml_path: str, mode='val'): 
    ''' '''
    # write to csv in model/metrics_log.csv or txt
    # check if exists then create, otherwise append. specify 'val' or 'test' set


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# Main function

def active_learning_loop(args):
    # Unpack user args 
    total_loop_iterations = args.num_loop_iters # default 3
    num_epochs = args.epochs_per_iter # default 5
    dataset = args.dataset # default 'preprocessed'
    #device = args.device # default 'cpu'

    # Get dataset path
    dataset_parent_dir = os.path.join("data",dataset,"frames")

    line = '<>'*20
    print('')
    print(line)
    print("Object detection model training")
    print(line, '\n')
    print("(Press Ctrl+C at any time to terminate this program.)")
    print("")

    # Setup paths and folders for runs, models depending on existing runs
    run_datetime, current_iter, run_path, model_path, run_iter_path, model_iter_path = get_run_model_paths(dataset_parent_dir)

    # Setup base model
    base_model_name = "yolo11n.pt"
    base_model_path = os.path.join("models/base", base_model_name)
    model = YOLO(base_model_path)

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
            print(line,'\n')
            break
        

        # Setup for training

        # Train base model on current Train set in iterationX folder
        # store run results in project=model/iterationX/ name=training
        train_model(yolo_model_object=model)
        # Evaluate on Validation set, same project, name=evaluation

        # rename current folder run_path/iteration{loop_iteration_num+1}, update variable name
        # create folder with old name, and only copy over labels/[train] ? minimal set no images anyway

        # Infer on unlabelled pool - store inference labels in data/temp/labels
        # if data/temp/labels is then empty, no detections for anything, break. Otherwise:
        # Query data/temp/labels, get high conf inferences list, move associated images list into Train, and copy over inference labels into Train
        # make list of low conf inferences. copy into data/temp/review_images
        # Human review of low confidence inferences from data/temp/review_images and data/temp/labels
        annotation_window_wrapper()
        # Now have correct labels
        # Move images into run_path/iterationX/images/train
        # Move labels into run_path/iterationX/labels/train
        # Now both folders should be empty, but check anyway
        # print statement about this is done
        loop_iterations_completed += 1
        print("Do you want to continue?")
        print(">>> Press Enter to continue, or enter 's' to skip to final training.")
        answer = input()
        if answer=='s':
            break


        
    
    # Train base model on final Train set (current iterationX), project=model/iterationX/ name=training
    # print final trained model weights .pt location
    # Evaluate model on Validation set like normal, project=model/iterationX/ name=evaluation

    # Plot validation map50-95 scores as function of iteration and train set size, save as jpg in model/iterationX

    # Evaluate model on Test set (different yaml), project=model/iterationX/ name=evaluation_test

    # Ask user if want to use final model on sample video from Test set
    # and if they want to compare base model and new or just new
    # List videos dictionary in Test set for user, and tell them to enter 1 for vid1, 2 for vid2 etc
    # get video path
    # ask if want to compare with base
    if loop_iterations_completed == 0:
        print("base model is final model, only predicting with final model")
    # inference with video, project=model/iterationX/ name=final_model_annotated_video
    # also if with base model and valid, do name=base_model_annotated_video

    # print final trained model weights .pt location

    # say done





if __name__=="__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_loop_iters', type=int, required=False, default=3, help="The number of active learning loop iterations. This is the number of times you will have to review machine annotations. (Default: 3)")
    parser.add_argument('--epochs_per_iter', type=int, required=False, default=5, help="The number of model training epochs performed in each iteration of the loop. (Default: 5)")
    parser.add_argument('--dataset', type=str, required=False, default='preprocessed', const='preprocessed', nargs='?', choices=['preprocessed', 'custom'], help="The dataset you are using, either 'preprocessed' or 'custom', depending on if you created your own dataset. (Default: preprocessed)")
    #parser.add_argument('--device', type=str, required=False, default='cpu', const='cpu', nargs='?', choices=['cpu', 'cuda', 'mps'], help="Device to perform model computations on, either 'cpu' for CPU, 'cuda' for GPU with CUDA compatibility, or 'mps' for Apple MPS on a Mac Metal GPU. (Default: 'cpu')")
    args = parser.parse_args()
    active_learning_loop(args)