import os
from pipeline_functions.annotation_window import annotation_window_wrapper

# Quick wrapper for annotation window with pre-selected set of images

def main():
    images_folder_to_annotate = "data/preprocessed/frames/images/val"
    annotations_folder = "data/temp/try_annotation_labels"
    os.makedirs(annotations_folder, exist_ok=True)

    annotation_window_wrapper(images_folder=images_folder_to_annotate,
                            labels_folder=annotations_folder)

if __name__=='__main__':
    main()
