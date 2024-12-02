import os
import argparse
from ultralytics import YOLO
from pipeline_functions.utils import *

'''
YOLO wrapper script to use pretrained model to infer on source media
'''

def main(args):
    # Unpack user args 
    source = args.source
    model_path = args.model
    output_dir = args.output_dir
    show_in_window = args.show_in_window
    show_labels = args.show_labels
    show_conf = args.show_conf
    verbose = args.verbose
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device # default 'cpu'

    # Get right device
    device = select_device(device=device)

    # Setup model
    try:
        model = YOLO(model_path)
    except:
        raise Exception(f"No valid model found at model path {model_path}.")
    model.to(device)

    classes=[32]

    line='<>'*20
    print(line)
    print("Using given model to perform object detection on media in source.")
    print(line,'\n')

    results = model.predict(source=source,
                            show=show_in_window, show_labels=show_labels, show_conf=show_conf,
                            max_det=1,
                            classes=classes,
                            save_txt=True, save_conf=True,save=True,
                            verbose=verbose,
                            project=output_dir, name="prediction",
                            exist_ok = True)
    print("")
    print(">>> Press Enter when done viewing prediction")
    answer = input()

    print("")
    print(line)
    print("Prediction complete!")
    print("You can find the predictions inside:")
    print(f"{output_dir}/prediction")
    print(line,'\n')
    


if __name__=="__main__":
    # Parse input arguments
    line="<>"*20
    description = line +'\n'+"predict.py input arguments. (Note this is a very basic script, can be easier to manually change in code.)" + line
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source', type=str, required=True, help='The path to the source media you want to perform object detection on. Can be a directory of images, a single image or a single video.')
    parser.add_argument('--model', type=str, required=False, default='models/base/yolo11x.pt', help="The path to the trained YOLO model's weights. (Default: models/base/yolo11m.pt)")
    parser.add_argument('--output_dir', type=str, required=False, default='models', help="The path to the directory you want to store the inference run and annotated media in. (Default: models/)")
    parser.add_argument('--show_in_window', type=bool, required=False, default=True, help="Whether to show newly annotated frames/videos as they are predicted in a seperate window. (Default: True)")
    parser.add_argument('--show_labels', type=bool, required=False, default=True, help="Whether to plot detected object name on annotated frames. (Default: True)")
    parser.add_argument('--show_conf', type=bool, required=False, default=True, help="Whether to plot detected object name prediction confidence scores on annotated frames. (Default: True)")
    parser.add_argument('--verbose', type=bool, required=False, default=True, help="Whether to print out detection message for each frame in source as model infers on it. (Default: True)")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="The batch size used during model inference. (Default: 16)")
    parser.add_argument('--num_workers', type=int, required=False, default=8, help="The number of worker threads for dataloading during model inference. (Default: 8)")
    parser.add_argument('--device', type=str, required=False, default='cpu', const='cpu', nargs='?', choices=['cpu', 'cuda', 'mps'], help="Device to perform model computations on, either 'cpu' for CPU, 'cuda' for GPU with CUDA compatibility, or 'mps' for Apple MPS on a Mac Metal GPU. (Default: 'cpu')")
    args = parser.parse_args()
    main(args)