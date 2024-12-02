# Video Object Detection - Model Training Pipeline

Nov 2024 \
*Ben Winstanley*

----

![football_annotated](https://github.com/benw000/Object-Detection/blob/dev/data/demo_vids/football_annotated.gif)

![annotation_window](https://github.com/benw000/Object-Detection/blob/dev/data/demo_vids/annotation_demo.gif)

### Pipeline to train and refine an object detection model on a custom sports ball dataset, complete with custom annotation window and extraction workflow.

----

## Usage:

*(Enter the following commands in your command line shell)*

### Setup
Download repository files: 
```
git clone https://github.com/benw000/Object-Detection.git
```

Set up a Python virtual environment to handle dependencies: 
```
chmod +x setup.sh
./setup.sh
```
(or manually):
```
python3 -m venv obj_det_venv
source obj_det_venv/bin/activate
pip install -r requirements.txt
```
### Preprocessing

Download a selection of sports videos from YouTube and extract frames from each:
```
python preprocess.py
``` 

Perform initial machine-assisted annotation on a subset of this data to provide Train, Test and Validation sets: 
```
python initial_annotation.py
```

### Training and Evaluation

Iteratively refine a pre-trained base model using an active learning loop:
> Here we specify 5 loop iterations, each with 10 training epochs, operating on the machine's GPU (if available).
```
python train.py --num_loop_iters 5 --epochs_per_iter 10 --device gpu
```
Use trained model to predict object detection annotations for a video:
> Here we specify the video path and the path of our trained model's weights, as well as an optional argument to see inference in real time on a window.
```
python predict.py --source data/demo_vids/football_original.mp4 --model model/base/yolo11m.pt --show_in_window true
```

### Try the Annotation Window!

Try out the annotation window outside of the main pipeline, labelling a preselected set of images from data/preprocessed:
```
python try_annotation.py
```

----

## Guide:

This is an iterative pipeline for improving object detection models, specifically to train them to better detect sports balls (footballs, basketballs, golfballs etc.)

#### Data
This project works with video data taken from the **[Sports-1M](https://github.com/gtoderici/sports-1m-dataset/tree/master)** dataset, a collection of 1 million YouTube videos which were tagged with a particular sport by the YouTube API. \
Users either work with a preprocessed dataset of videos, containing ~1200 human-labelled images and ~3000 unlabelled, or they construct their own custom dataset. This is done in the preprocessing scripts by downloading a number of videos from the Sports-1M set, extracting frames from these, and then manually labelling a subset of those frames.

#### Active Learning Loop

This pipeline seeks to address the challenge of training robust models when access to labelled data is quite limited. By employing an **Active Learning Loop**, the pipeline allows users to provide a minimal amount of human annotation, while maximizing model improvement at the same time as building a set of trusted pseudo-labels.

This involves the following steps:
1) Produce an initial dataset, with small labelled *Train*, *Validation* and *Test* subsets, and a larger *unlabelled pool* of images.
2) Import a pre-trained general Object Detection Model as our *base model*. We use the **[YOLOv11](https://docs.ultralytics.com/models/yolo11/)** model from **ultralytics**.
3) Repeat N many times:
    1) Refine our *base model* by training on our *Train* set, and evaluating on the *Validation* set.
     Use the refined model to predict annotations for the images in the *unlabelled pool*, each annotation detection having its own confidence score in [0,1]
    2) Automatically accept any annotations with a confidence score above a certain threshold. These are pseudo-annotations, produced by our model, which we choose to remove from the *unlabelled pool* and add to the *Train* set.
    3) For any remaining annotations above a lower threshold, instruct the human user to review and correct a subset of them, and then similarly add them to the *Train* set. This focusses the human user's labelling efforts on lower confidence detections, where a human's input is needed more.
4) Finally, train our *base model* on the final *Train* set, now much larger in size as it contains many pseudo-labels. Evaluate the model on the *Validation* set, as well as the *Test* set to produce final evaluation scores.

----

### Extensions and todo:

- Work with better quality dataset - most videos in Sports-1M are either not available, old, grainy, low resolution, or irrelevant.
- Add the ability for the user to pick up an **Active Learning Loop** from where they left off in a previous run.
- Display a histogram of confidence weights after each inference, so that the user might change confidence thresholds in response to the distribution of confidences.
- Investigate alternative models to YOLO which incorporate temporal tracking between similar frames, in order to learn trends like physical momentum of a ball within a video.
- Produce versions of each file that can be automated up until human labelling.
- Develop code to deploy trained models to detect objects in an incoming video stream.
- Further bug fixes and cleanup.

Currently, refine training the imported model seems to wipe its knowledge and ability to detect any instances at all - after training on the initial *Train* set, inferring on the *unlabelled pool* set produces 0 detections, when ~200 are expected. I've tried freezing the NN layers up to the backbone, and even all the way up to the detection head of the YOLO model. I've also tried massively reducing the learning rates to 10e-6 or less, but to no avail. This will take further investigation, and I'm short on time. 
The active learning loop is still useful if the training step is removed, and instead, base model inferences are used to grow the Train set in conjunction with human-reviewed annotations.

----

### Files

#### Scripts:
- setup.sh 
- preprocess.py
- innitial_annotation.py
- train.py
- predict.py
- try_annotation.py
- pipeline_functions/annotation_window.py : *contains main code for annotation window.*
- pipeline_functions/utils.py : *contains various functions used in the repo.*

#### Preprocessed dataset:
- data/preprocessed/videos : *contains small number of Sports-1M videos, hand-picked to be decent quality, divided into [train, val, test] subfolders.*
- data/preprocessed/downloads_log.json : *contains log of YouTube downloads for data/preprocessed/videos.*
- data/preprocessed/frames : *general preprocessed dataset for model training, containins [images, labels] subfolders.*
- data/preprocessed/frames/images : *contains extracted frames from videos, divided into [train, val, test] subfolders, each containing frame_00xxx.jpg files.*
- data/preprocessed/frames/labels : *contains corresponding annotation files, divided into [train, val, test] subfolders, each containing frame_00xxx.txt files, with annotations in YOLO format.*

#### Other data:
- data/sports_1m : *contains the .txt files downloaded from the Sports-1M GitHub.*
- data/custom : *same folder structure as data/preprocessed, but empty.*
- data/temp : *empty temporary cache folder for script operations.*
- data/demo_vids : *contains the gifs seen above.*

#### Misc:
- requirements.txt : *packages needed to set up a python virtual environment capable of running YOLO training.*
- .gitignore : *tells GitHub what to ignore.*


----

### Licenses:

**YOLOv11** by **ultralytics**: \
available under a [GNU Affero General Public License v3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 

**Sports-1M** by **Karpathy et al**: \
available under a [Creative Commons License v3.0](https://github.com/gtoderici/sports-1m-dataset/blob/master/LICENSE.md)

----



