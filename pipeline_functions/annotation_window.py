import cv2
import os
import time
import tkinter
import argparse
import subprocess
import numpy as np

# Initialise globals - using global variables instead of passing to functions, since cv2 functions are weird
drawing = False
start_x, start_y = -1, -1
pos_x, pos_y = -1, -1
current_image = None
image_height, image_width = 300,300
screen_width, screen_height = 1000,1000
class_id = 32
bboxes = [] # These are stored in a stack datastructure, so most recent can be popped

# Easter egg!
# Sound effects - set to True and attach paths to sound effects during annotation to make things more entertaining :)
play_sound_effects = False
done_sound = "experimentation/sounds/blink.m4a"
undo_sound = "experimentation/sounds/boowomp.wav"
quit_sound = "experimentation/sounds/spongebob-fail.wav"
thanks_sound = "experimentation/sounds/thank-you-for-your-patronage.wav"

'''
Helper functions
'''
def annotation_window_wrapper(images_folder, labels_folder):
    '''
    Function to executes this current script in command line,
    with arguments passable by other python scripts.
    '''
    command = [
        "python", __file__,
        "--img_dir", images_folder,
        "--label_dir", labels_folder
    ]
    subprocess.run(command)


def maximize_window(window_name):
    """
    Maximizes the OpenCV window to fit the screen while leaving space for the menu bar.
    """
    # Resize the OpenCV window to match screen dimensions minus a buffer
    buffer = 50  # Leave some space for menu bars and borders
    cv2.resizeWindow(window_name, screen_width - buffer, screen_height - buffer)


def draw_bboxes(img, bboxes):
    '''
    Iterate through bboxes and plot onto "img".
    '''
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def mouse_callback(event, x, y, flags, param):
    '''
    Mouse monitoring function, reacts to mouse events.
    '''
    # Access global variables
    global drawing, start_x, start_y, current_image, pos_x, pos_y

    # Update mouse position
    pos_x, pos_y = x, y

    # Copy 'current_image' into temporary 'img_copy'
    # We plot transient mouse effects onto 'img_copy' and not the main image
    img_copy = current_image.copy()

    annotation_done = False
    # Mouse movement
    if event == cv2.EVENT_MOUSEMOVE: 
        if not drawing:
            # Draw crosshair
            crosshair_size = 40
            color = (0, 255, 255)  # yellow
            thickness = 1
            cv2.line(img_copy, (x - crosshair_size, y), (x + crosshair_size, y), color, thickness)
            cv2.line(img_copy, (x, y - crosshair_size), (x, y + crosshair_size), color, thickness)
        else:
            # Draw bbox rectangle being created if in draw mode
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (255, 0, 0), 2)

    # Mouse left click down
    if event == cv2.EVENT_LBUTTONDOWN: 
        # Start drawing a new bounding box
        drawing = True
        start_x, start_y = x, y
        # Plot the zero area bbox as dot upon clicking
        cv2.rectangle(img_copy, (start_x, start_y), (x, y), (255, 0, 0), 2)

    # Mouse left click release
    if event == cv2.EVENT_LBUTTONUP: 
        # Finish drawing the bounding box
        drawing = False
        end_x, end_y = x, y
        # Append the bounding box to the list, ensuring top-left and bottom-right coordinates
        bboxes.append([min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)])
        annotation_done = True
            
        
    # Draw existing bounding boxes onto temporary image
    draw_bboxes(img_copy, bboxes)
    # Render the temporary image in our window display, with all objects plotted
    cv2.imshow(window_title, img_copy)
    if annotation_done and play_sound_effects:
        cmd = ["afplay", done_sound]
        subprocess.Popen(cmd)


def redraw_display():
    '''
    Redraw current display with bboxes, crosshair or dynamic bbox creation.
    (Same as mouse_callback but without needing a mouse event update).
    '''
    global drawing, start_x, start_y, current_image, pos_x, pos_y

    img_copy = current_image.copy()
    if not drawing:
        # Draw crosshair
        crosshair_size = 40
        color = (0, 255, 255)  # yellow
        thickness = 1
        cv2.line(img_copy, (pos_x - crosshair_size, pos_y), (pos_x + crosshair_size, pos_y), color, thickness)
        cv2.line(img_copy, (pos_x, pos_y - crosshair_size), (pos_x, pos_y + crosshair_size), color, thickness)
    else:
        # Draw bbox rectangle being created if in draw mode
        cv2.rectangle(img_copy, (start_x, start_y), (pos_x, pos_y), (255, 0, 0), 2)

    draw_bboxes(img_copy, bboxes)
    cv2.imshow(window_title, img_copy)


def save_annotations(image_path, labels_folder):
    '''
    Write current bbox annotations to txt file
    '''
    # Isolate 'image_name' string from 'images_folder/image_name.jpg' image_path
    image_name = os.path.splitext(os.path.basename(image_path))[0] 
    # Construct corresponding 'labels_folder/image_name.txt' path
    annotation_path = os.path.join(labels_folder, image_name+'.txt')
    # If bboxes empty then remove the file and do nothing
    if bboxes==[]:
        if os.path.exists(annotation_path):
            os.remove(annotation_path)
        return
    # Ensure path exists, then write boxes to it line by line
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    with open(annotation_path, "w") as f:
        for x1, y1, x2, y2 in bboxes:
            # Convert to normalised coordinates before writing
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            # Save in YOLO format: class_id x_center y_center width height
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def load_annotations(image_path, labels_folder):
    '''
    Load in bbox annotations from a .txt file. 
    '''
    # Isolate 'image_name' string from 'images_folder/image_name.jpg' image_path
    image_name = os.path.splitext(os.path.basename(image_path))[0] 
    # Construct corresponding 'labels_folder/image_name.txt' path
    annotation_path = os.path.join(labels_folder, image_name+'.txt')
    # If the path exists, read in bboxes
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # No confidence value
                    _, x_center, y_center, width, height = map(float, parts)
                elif len(parts) == 6:  # Includes confidence value
                    _, x_center, y_center, width, height, conf = map(float, parts)
                else:
                    raise ValueError(f"Invalid line format in {annotation_path}: {line}")
                # Convert from YOLO format to pixel coordinates
                x1 = int((x_center - width / 2) * image_width)
                y1 = int((y_center - height / 2) * image_height)
                x2 = int((x_center + width / 2) * image_width)
                y2 = int((y_center + height / 2) * image_height)
                bboxes.append([x1, y1, x2, y2])


def get_supported_images(folder):
    ''' 
    Get list of image file paths from folder
    '''
    supported_formats = {".png", ".jpg", ".jpeg", ".bmp"}
    files_list = os.listdir(folder)
    image_paths_list = []
    for file in files_list:
        if os.path.splitext(file)[1].lower() in supported_formats:
            image_paths_list.append(os.path.join(folder, file))
    return image_paths_list


























'''
Main Script
'''

if __name__=="__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--label_dir', type=str, required=True)
    args = parser.parse_args()
    images_folder = args.img_dir # z_delete
    labels_folder = args.label_dir  # z_delete_labels

    # Get screen dimensions (for later maximising)
    root = tkinter.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Get list of supported images in the folder
    image_paths_list = get_supported_images(images_folder)
    if not image_paths_list:
        print("No images found in the folder!")
        exit()
    num_images = len(image_paths_list)

    # Loop over images and annotate each
    for index, image_path in enumerate(image_paths_list):
        name = os.path.basename(image_path)
        # Copy image into a cv2 matrix and get dimensions
        current_image = cv2.imread(image_path)
        image_height, image_width = current_image.shape[:2]

        # Set up a cv2 window to annotate inside
        if index==0:
            # Set up new window at start
            window_title = f"{name} ({index+1}/{num_images}) u: Undo last | n: Next (+Save) | q: Quit (+Save)"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL) # window normal allows resizing to full screen
        else: 
            # Rename window title
            new_title = f"{name} ({index+1}/{num_images}) u: Undo last | n: Next (+Save) | q: Quit (+Save)"
            cv2.setWindowTitle(window_title, new_title)
            # window_title = new_title

        # Display image and maximise to full screen
        cv2.imshow(window_title, current_image)
        maximize_window(window_title)

        # Set the mouse callback function for this window
        cv2.setMouseCallback(window_title, mouse_callback)

        # Load existing bboxes into list, then draw them in starting window
        bboxes = []
        load_annotations(image_path, labels_folder)
        redraw_display()

        # Main response loop (cv2 key stuff taken from online examples)
        while True:
            # Get the key that is pressed, (with built in delay)
            delay = 1 # delay in milliseconds between key press events
            key = cv2.waitKey(delay=delay) & 0xFF

            # u: Undo
            if key == ord("u"):
                # Undo last bounding box by popping from stack
                if bboxes:
                    bboxes.pop()
                # Refresh window display with updated bboxes
                redraw_display()
                if play_sound_effects:
                    cmd = ["afplay", undo_sound]
                    subprocess.Popen(cmd)

            # n: Save + next image
            elif key == ord("n"):
                # Save current bbox annotations to txt file
                save_annotations(image_path, labels_folder)
                # Break current image's loop, progress to next
                break
            
            # q: Save + quit
            elif key == ord("q"):
                # Save current bbox annotations to txt file
                save_annotations(image_path, labels_folder)
                # Destroy all cv2 windows
                cv2.destroyAllWindows()
                # Terminate script
                print(f"{index+1}/{num_images} images in {images_folder} annotated, stored in {labels_folder}.")
                if play_sound_effects:
                    cmd = ["afplay", quit_sound]
                    subprocess.Popen(cmd)
                exit()
    # Finished looping over images, close window
    cv2.destroyAllWindows()
    # done sound
    if play_sound_effects:
            cmd = ["afplay", thanks_sound]
            subprocess.Popen(cmd)
    # End print message
    print(f"All images in {images_folder} successfully annotated, annotations stored in {labels_folder}.")
  