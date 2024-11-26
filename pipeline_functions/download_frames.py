import os
import csv

'''
File containing function to download custom dataset from the Sports-1m dataset.
'''

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

    print("Available videos:")
    print(sport_counts)

def download_youtube_videos(max_vids: int):
    from pytubefix import YouTube
    from pytubefix.cli import on_progress
    import time
    import random
    import json
    
    output_path = 'data/custom/videos/'
    videos_list_path = 'data/sports_1m/train_partition_filtered.txt'

    file = open(videos_list_path, 'r')

    success_count = 0
    for line in file:
        if success_count > max_vids:
            return
        success_count += 1
        # Get URL from each line of the filtered URL list.
        url = str(line)
        try:
            # Download video
            yt = YouTube(url, on_progress_callback=on_progress)
            print(success_count, ":", yt.title)
            ys = yt.streams.get_highest_resolution()
            ys.download(output_path = output_path)
            print("Downloaded. Waiting 5 seconds before next download.")
            time.sleep(5)
            success_count += 1
        except:
            print(f"URL {url} failed. Trying next URL.")
    file.close()
    return

def download_custom_frames():
    '''
    Main function to download custom dataset.
    Asks user for details at each stage.
    '''
    line = '<>'*20
    print('')
    print(line)
    print("Download options")
    print(line, '\n')

    print("The Sports-1M dataset contains URLs to around ~1 million sports videos on YouTube.")
    print("Each URL is labelled with a number of sport names, a list of these is found in >data/sports_1m/labels.txt ")
    print("Please edit >data/sports_1m/my_custom_classes.txt so that it contains a subset of ball sports from the list of available sports.")
    print(">")
    while True:
        print("Press Enter when my_custom_classes.txt is edited and saved.")
        _ = input()
        flag, label_dict = check_custom_classes_valid()
        if flag==0:
            break

    print("How many videos total do you want to download?")
    print(">")
    print("Please enter a positive number, or 'n' to cancel custom dataset process.")
    while True:
        answer = input()
        if answer=='n':
            print("Using default preprocessed dataset.")
            return 1
        else:
            try:
                num = int(answer)
                if num <= 0:
                    pass
                else:
                    break
            except:
                pass
    
    # Get list of valid videos
    produce_video_list(label_dict)

    print("\n")
    print("DISCLAIMER: This script uses the 'pytubefix' Python module to try to download the YouTube videos. (See download_youtube_videos() function in >pipeline_functions/download_frames.py)")
    print("The large number of download requests may cause YouTube to block your machine's IP from accessing YouTube again.")
    print("Alternatively, you can manually download each video and place it inside >data/custom/videos.")
    print("By confirming, you acknowledge this risk and take responsbility for anything that might happen.")
    while True:
        print("Please enter 'y' to go ahead with this, 'n' to cancel and use the preprocessed dataset, or 'w' if you want to manually download the videos.")
        answer = input()
        if answer=='n':
            print("Using default preprocessed dataset.")
            return 1
        elif answer=='y':
            print("Downloading videos from list...")
            download_youtube_videos(max_vids=num)
            break
        elif answer=='w':
            print("Opted for manual download.")
            print("Press Enter when finished.")
            _ = input()
            break
        else:
            pass

    # TODO: make another git commit with description of progress
    # now videos are downloaded, extract raw frames into 1 single folder
    # ask user how many frames per vid


    return 0