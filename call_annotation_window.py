# Example script to successfully call annotation_window.py !
# so happy this works lol

from pipeline_functions.annotation_window import annotation_window_wrapper

images_folder = "z_delete"
labels_folder = "z_delete_labels"

annotation_window_wrapper(images_folder=images_folder, labels_folder=labels_folder)