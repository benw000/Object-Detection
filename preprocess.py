import os
import csv
import math
import time
import json
import shutil
import random
import subprocess
from pytubefix import YouTube
from pytubefix.cli import on_progress
    
'''
File containing functions to download custom dataset from the Sports-1m dataset.
'''

# Helper functions 

def get_positive_number(max_val=None):
    while True:
        answer = input()
        try:
            num = int(answer)
            if num <= 0:
                print("Please enter a positive, nonzero value.")
                pass
            elif (not max_val is None) and num >= max_val:
                print("Number too high. Please enter smaller value")
                pass
            else:
                break
        except:
            print("Input error")
            pass
    return num

def check_custom_classes_valid():
    labels_path = 'data/sports_1m/labels.txt'
    custom_labels_path = 'data/sports_1m/my_custom_classes.txt'
    try:
        # Read all labels
        with open(labels_path, 'r') as f:
            labels_list = []
            for line in f:
                labels_list += [line.rstrip()]
        with open(custom_labels_path, 'r') as f:
            custom_labels_list = []
            for line in f:
                custom_labels_list += [line.rstrip()]
            if custom_labels_list==[]:
                f"Please specify some ball sports, {custom_labels_path} is currently empty."
                return 1, None
        # Check valid, and get label dictionary
        label_dict = {}
        for label in custom_labels_list:
            if not label in labels_list:
                print(f"Custom label '{label}' is invalid (does not appear in main list).")
                return 1, None
            class_index = labels_list.index(label)
            label_dict[class_index] = label
        return 0, label_dict # if all valid
    except:
        print(f"Please ensure {labels_path} and {custom_labels_path} both exist and are in the correct format.")
        return 1, None
    
def produce_video_list(label_dict: dict):
    # Read a training set and export line into new file if sports id matches
    old_path = 'data/sports_1m/train_partition.txt'
    new_path = 'data/sports_1m/train_partition_filtered.txt'

    old_file = open(old_path, 'r')
    new_file = open(new_path, 'w')

    # Keep track of a count for each sport
    vid_count = 0
    sport_counts = {}

    for line in old_file:
        # Find the end of the youtube URL, isolate numbers at end
        space = line.index(' ')
        csv_str = line[space+1:]
        # Parse numbers with csv reader, write new line if sports id in list
        for row in csv.reader([csv_str], delimiter=','):
            num = int(row[0])
            if num in label_dict.keys():
                # Write URL only to new file
                new_file.write(line[:space]+'\n')
                # Update counts
                count = sport_counts.get(label_dict[num], 0)
                sport_counts[label_dict[num]] = count+1
                vid_count += 1

    old_file.close()
    new_file.close()

    print("Number of available videos in Sports-1M train dataset:")
    print(sport_counts)


def download_youtube_videos(max_vids: int, cooldown_range):
    # Downloads youtube videos from list, with random cooldown between download attempts
    # sampled uniformly from the cooldown_range

    # Unpack cooldown_range
    lower, upper = cooldown_range[0], cooldown_range[1]

    # Paths
    output_path = 'data/custom/videos'#'data/custom/videos/'
    videos_list_path = 'experimentation/custom/valid.txt' #'data/sports_1m/train_partition_filtered.txt'
    json_path = 'data/custom/downloads_log.json' #'data/custom/downloads_log.json'

    # Start log for download info    
    download_log_list = []
    count, success_count = 1, 0

    # Loop through videos in train_partition_filtered.txt
    file = open(videos_list_path, 'r')
    for line in file:
        # Stop when max_vids is reached
        if success_count > max_vids:
            return

        # Get URL from each line of the filtered URL list.
        url = str(line.rstrip())

        # Initialise log entry
        entry_dict = {"attempt": count, "success": False, "url": url, "title": None, "length_seconds": None}
        try:
            print(f"\nStarting download attempt {count}: URL = {url}")

            # Initialise YouTube object
            yt = YouTube(url, on_progress_callback=on_progress)
            entry_dict["title"] = yt.title
            entry_dict["length_seconds"] = yt.length
            ys = yt.streams.get_highest_resolution()

            # Attempt download
            print(f"Downloading: {yt.title} ({yt.length // 60}:{yt.length % 60})...")
            ys.download(output_path = output_path)

            entry_dict["success"] = True
            success_count += 1
            print("Download successful.")
        except:
            print(f"Download failed. Trying next URL.")

        # Export current log to JSON
        download_log_list.append(entry_dict)
        with open(json_path, 'w') as f:
            json.dump(download_log_list, f, indent=4)

        # Randomized cooldown after each attempt between lower and upper
        cooldown = random.uniform(lower, upper)
        print(f"Cooldown: {cooldown:.2f} seconds...")
        time.sleep(cooldown)
        count += 1

    # Close URLs txt file
    file.close()
    print(f"\nDownloads log exported to {json_path}")
    return


def divide_videos(folder_path, num_train, num_test, num_val):
    # Divides videos into train, test and val folders

    # Paths for subfolders
    train_path = os.path.join(folder_path, 'train')
    test_path = os.path.join(folder_path, 'test')
    val_path = os.path.join(folder_path, 'val')

    # Ensure subfolders exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Get all MP4 files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') and os.path.isfile(os.path.join(folder_path, f))]
    random.shuffle(files)  # Shuffle to ensure randomness

    # Check if there are enough files
    total_required = num_train + num_test + num_val
    if len(files) < total_required:
        print(f"Error: Not enough MP4 files in the folder. Found {len(files)}, required {total_required}.")
        return

    # Distribute files into subfolders
    train_files = files[:num_train]
    test_files = files[num_train:num_train + num_test]
    val_files = files[num_train + num_test:num_train + num_test + num_val]

    # Move files to respective subfolders
    for f in train_files:
        shutil.move(os.path.join(folder_path, f), os.path.join(train_path, f))
    for f in test_files:
        shutil.move(os.path.join(folder_path, f), os.path.join(test_path, f))
    for f in val_files:
        shutil.move(os.path.join(folder_path, f), os.path.join(val_path, f))


def extract_frames_batch(video_path, output_folder, num_frames, running_frame_count):
    """
    Extract equally spaced frames from a video and save them with unique names in the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create a temporary folder to store all frames
    temp_folder = os.path.join(output_folder, "temp_frames")
    os.makedirs(temp_folder, exist_ok=True)

    # Extract all frames from the video with call to ffmpeg
    temp_output_pattern = os.path.join(temp_folder, "frame_%04d.jpg")
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path, "-vsync", "vfr", "-q:v", "2",
        temp_output_pattern, "-loglevel", "error"
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # List all frames in the temp folder
    all_frames = sorted(os.listdir(temp_folder))
    total_frames = len(all_frames)

    if total_frames < num_frames:
        selected_indices = range(total_frames)  # Use all frames if fewer than requested
    else:
        step = total_frames / num_frames
        selected_indices = [math.floor(i * step) for i in range(num_frames)]

    # Copy the selected frames to the output folder with unique names
    for index in selected_indices:
        frame_file = all_frames[index]
        src_path = os.path.join(temp_folder, frame_file)
        dest_path = os.path.join(output_folder, f"frame_{running_frame_count:06d}.jpg")
        os.rename(src_path, dest_path)
        running_frame_count += 1

    # Cleanup temporary folder
    for f in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, f))
    os.rmdir(temp_folder)

    return running_frame_count


def extract_frames(videos_path, folder_path, num_frames_train, num_frames_test, num_frames_val):
    """
    Process videos in 'train', 'test', and 'val' subsets sequentially, extracting equally spaced frames
    and saving them with unique names.
    """

    subsets = {
        "train": num_frames_train,
        "test": num_frames_test,
        "val": num_frames_val,
    }

    for subset, num_frames in subsets.items():
        print(f"Processing {subset} subset:")
        input_folder = os.path.join(videos_path, subset)
        output_folder = os.path.join(folder_path, subset)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get all video files in the input folder
        video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

        running_frame_count = 0  # To ensure unique frame names across all videos in the subset

        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)
            print(f"Video: {video_file}")
            running_frame_count = extract_frames_batch(
                video_path, output_folder, num_frames, running_frame_count
            )
        print("\n")


def move_images(source_folder, destination_folder, num_images):
    '''
    Move N images from Train set into Unlabelled Pool
    '''
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    images = os.listdir(source_folder)

    # Move the files
    for i in range(num_images):
        source_path = os.path.join(source_folder, images[i])
        destination_path = os.path.join(destination_folder, images[i])
        shutil.move(source_path, destination_path)




# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------















# Main frame extraction pipeline function

def download_custom_frames():
    '''
    Main function to download custom dataset.
    Asks user for details at each stage.
    '''
    line = '<>'*20

    # Download Options
    print('')
    print(line)
    print("Preprocessing: Download options")
    print(line, '\n')

    print("(Press Ctrl+C at any time to terminate this program.)")
    print("The Sports-1M dataset contains URLs to around ~1 million sports videos on YouTube.")
    print("Each URL is labelled with a number of sport names, a list of these is found in < data/sports_1m/labels.txt > ")
    print("Please edit < data/sports_1m/my_custom_classes.txt > so that it contains a subset of ball sports from the list of available sports.")
    print("")
    while True:
        print(">>> Press Enter when < my_custom_classes.txt > is edited and saved.")
        _ = input()
        flag, label_dict = check_custom_classes_valid()
        if flag==0:
            break

    # Get list of valid videos
    produce_video_list(label_dict)

    # Ask for number of videos wanted
    print("")
    print(">>> How many videos total do you want to download? (Please enter a positive number).")
    num_videos = get_positive_number()
    
    # Warn user then proceed with downloading videos
    print("\n")
    print("This script uses the 'pytubefix' Python module to attempt to download the YouTube videos.")
    print("You will specify a time range (e.g. 30 secs - 60 secs) which we randomly sample from to produce a cooldown period, used between each download request to avoid bot detection.")
    print("DISCLAIMER: There is a small risk that the large number of download requests from a single IP may cause YouTube to temporarily block your machine's IP.")
    print("(See download_youtube_videos() function in  < pipeline_functions/download_frames.py > for implementation.)")
    print("")
    print("Alternatively, you can manually download videos listed in < data/sports_1m/train_partition_filtered.txt > and place it inside  < data/custom/videos >. Some of the URLs are to videos that have been taken down.")
    print("")
    while True:
        print(">>> Please enter:")
        print("   'y' to go ahead and use this script to download the videos,")
        print("   'm' if you want to manually download the videos / have them downloaded already,")
        print("    or 'n' to terminate the script (and use the already processed dataset).")
        answer = input()
        if answer=='n':
            print("Terminating.")
            return 1
        elif answer=='y':
            # Downloading with pytube, get cooldown upper and lower bounds
            print(">>> Please specify integer lower and upper bounds (in seconds) for the cooldown time. (80-100 seconds recommended to avoid all bot detection.)")
            while True:
                print("Lower bound:")
                lower = get_positive_number()
                print("Upper bound:")
                upper = get_positive_number()
                if upper>lower:
                    break
                else:
                    print("Upper bound must be greater than lower bound.")
            print("Downloading videos from list...")
            download_youtube_videos(max_vids=num_videos, cooldown_range=[lower, upper])
            break
        elif answer=='m':
            print("Opted for manual download.")
            print("")
            print(">>> Press Enter when finished downloading videos.")
            _ = input()
            break
        else:
            pass
    
    # Make user check downloaded videos
    print("Please take a moment to review the downloaded videos and remove any irrelevant/dud videos.")
    print("")
    print(">>> Press Enter when done.")
    _ = input()

    # Frame extraction
    print('')
    print(line)
    print("Preprocessing: Extracting Frames")
    print(line, '\n')

    
    videos_dir = "data/custom/videos"
    files = [x for x in os.listdir(videos_dir) if not x.startswith('.')]

    num_vids = len(files)

    # Get train : validation : test  split from user
    print(f"{num_vids} videos are available in the dataset.")

    print("We will now split this dataset into different subsets:")
    print("    -Train: Used to train the sports ball detection model,")
    print("    -Validation: Used to monitor model performance during training (not used in training),")
    print("    -Test: Used to evaluate the final model's performance")

    print("We will assign each video into one of these subsets and extract a number of frames from them.")
    print("Frames from the Validation and Test sets will be manually annotated")
    print(" - they will first be machine-annotated by our base model (YOLOv11), before being passed to the user for manual reviewing and correction.")

    print("A number of frames will be selected from the Train set and manually annotated to form an Initial Train set.")
    print("The rest of the Train set frames will form an unlabelled pool of frames which we will draw from during active learning.")
   
    print("")
    print("You now need to specify:")
    print("1) How many videos do you want in then Validation and Test sets (remaining go in Train set).")
    print("2) Inside each subset, how many frames should be extracted per video (more frames means more annotating).")
    print("3) How many frames from the Train set do you want in the Initial Train set (these will need manually annotating). ")
    print("")

    print("1) Video counts: [Recommended ratio: Train - 80%, Validation - 10%, Test - 10%]")
    print(">>> How many videos do you want in the Test set (for final model evaluation)?")
    num_test = get_positive_number(max_val=num_vids/2)
    print(">>> How many videos do you want in the Validation set (for testing during training)?")
    num_valid = get_positive_number(max_val=num_vids-num_test)
    num_train = num_vids-num_test-num_valid
    print(f"Video counts: Train - {num_train}, Validation - {num_valid}, Test - {num_test}.")
    print("")

    # Randomly divide up videos into folders
    divide_videos(videos_dir, num_train=num_train, num_val=num_valid, num_test=num_test)

    print("2) Number of frames to extract:")
    print(f">>> How many frames should be extracted from each of the {num_train} Train videos? [Recommended ~200]")
    num_frames_train = get_positive_number()
    print(f">>> How many frames should be extracted from each of the {num_valid} Validation videos? [Recommended ~20-50]")
    num_frames_valid = get_positive_number()
    print(f">>> How many frames should be extracted from each of the {num_test} Test videos? [Recommended ~20-50]")
    num_frames_test = get_positive_number()
    total_train, total_valid, total_test = num_train * num_frames_train ,num_valid * num_frames_valid ,num_test * num_frames_test ,
    print(f"Total frames: Train - {total_train}, Validation - {total_valid}, Test - {total_test}.")
    print("")

    print("3) Initial Train set size:")
    print(">>> How many frames from the Train set do you want in the Initial Train set? [Recommended ~500]")
    total_initial_train = get_positive_number(max_val=total_train/2)

    # Extract frames from these videos
    print("")
    print("Extracting frames from each video:")
    frames_dir = "data/custom/frames/images"
    extract_frames(videos_dir, frames_dir, num_frames_train, num_frames_valid, num_frames_test)

    # Move frames from Train set into new unlabelled pool folder
    total_unlabelled = total_train - total_initial_train
    src = "data/custom/frames/images/train" 
    dest = "data/custom/frames/images/unlabelled"
    move_images(source_folder=src, destination_folder=dest, num_images=total_unlabelled)
        
    print('')
    print(line)
    print("Preprocessing finished!")
    print(line, '\n')
    print("Please now run <initial_annotation.py> to generate bounding box annotations for the dataset you've created.")
    print("")
    print("Run:")
    print("python initial_annotation.py")
    print("")
    return 0

if __name__=="__main__":
    download_custom_frames()