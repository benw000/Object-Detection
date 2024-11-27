import os
from pipeline_functions import *

'''
Main python script to control object detection pipeline.
'''

def main():
    '''
    Main pipeline function
    '''
    # Establish if the user wants to use preprocessed dataset, 
    # or download their own dataset from sports-1m:
    line = '<>'*20
    print('')
    print(line)
    print("Preprocessing data")
    print(line, '\n')
    print("(Press Ctrl+C at any time to terminate program.)")
    print("This pipeline can either use a preprocessed dataset, with ~5000 pseudo-annotated images and ~50 human-annotated test images,")
    print("or you can download and annotate a custom dataset from the Sports-1M dataset.")
    print("")
    print(">>> Do you want to use the preprocessed dataset? (Please answer 'y' or 'n')")
    while True:
        answer = input()
        if answer=='y':
            dataset_path = "data/preprocessed/frames"
            break
        elif answer=='n':
            # Use download_custom_frames from download_frames.py
            flag = download_custom_frames()
            if flag==1:
                dataset_path = "data/preprocessed/frames"
                break

            dataset_path = "data/custom/frames"
            # Annotate frames 
            
            break
    
    print(dataset_path)




if __name__=="__main__":
    main()
