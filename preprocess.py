import os
from pipeline_functions import *

'''
Python script to preprocess the Sports-1M dataset for use in object detection.
'''

def main():
    '''
    Main preprocessing pipeline
    '''
    line = '<>'*20
    print('')
    print(line)
    print("Preprocessing data")
    print(line, '\n')
    print("(Press Ctrl+C at any time to terminate program.)")
    print("You can either use the already preprocessed dataset, with ~5000 pseudo-annotated images and ~50 human-annotated test images,")
    print("or you can create a custom subset of the Sports-1M dataset and preprocess it with this script.")
    print("")
    print(">>> Enter 'y' to go ahead with custom dataset creation, or 'n' to terminate this script.")
    while True:
        answer = input()
        if answer=='n':
            print("Terminating.")
            return
        elif answer=='y':
            # Use download_custom_frames from download_frames.py
            flag = download_custom_frames()
            if flag==1:
                return
            break
        else:
            print("Invalid response, please enter 'y' or 'n'.")
    
    # Use pseudo_annotation_manager from pseudo_annotation.py
    pseudo_annotation_manager()

if __name__=="__main__":
    main()
