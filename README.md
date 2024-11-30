details

design principles:
-mention that supplied dataset is fundamentally bad quality, so not going to get great model from it
-explain different ways dataset bad
-would need massive model and dataset with lots of compute to train something that can cut through noise
-this is more a proof of concept
-therefore not expecting it to perform significantly better than standard YOLO, especially given tiny dataset size, and fact that its training on its own data.
-it may get better with active loop, but theres confirmation bias if psuedoannotation built on same model. fundamentally need humans to act as ground truth, lots of science to this.
-only improvement in quality is when humans introduce new training examples
-small scale
-uses lightweight models with quick training time
-quick end-to-end demonstration, would need more work to be rigorous, lacking in several areas
-most complicated bit of pipeline is the dataset handling
-the object detection is easy with SOTA models like YOLO and good quality datasets
-exercise in extracting the right dataset
-wanted to focus on nice user experience to simplify dataset creation from youtube videos

-urge them to try preprocessing to download 5 videos, with say 30 second cooldown, (put some work into this)
-explain used cvat but needed credentials, so made own annotation tool




Dataset flow:
-extract N~100 videos from internet
-divide N into {Unlabelled:70, Train: 10, Test: 10, Valid: 10}
-Split each subset into 200 frames per video. (Splitting by video means different context, more fair controlled test.)
-Generate human annotations for Train, Test and Valid: pass through YOLO then human review
-Hold off Valid and Test set, these are never trained on.
-Load in current ML model as downloaded YOLOv11: original_model
-Start active learning loop:
    we build Train set from slowly taking from Unlabelled set with pseudo+human
    -train original_model on Train set (we train from original each time!)
    -infer with current_model on entire Unlabelled set
    -take most confident conf>0.8 predictions and move these into Train set (pseudo annotations)
    -take least confident conf<0.5 predictions and human review subset of them, move into Train set (human annotations)
    Unlabelled set shrinks while Train set grows from pseudo+human annotations
    -report objective model accuracy by evaluating on Valid set, do this each loop iteration.
-Loop is over, train original_model on most recent Train set, this becomes final_model
-Report final_model accuracy by evaluating on Test set
-Visualise final model on a held back MP4 in the Test set, plus some frames from the Test set.


Utils we will need:

-initial human annotation: take frames folder, then for each of train, test, valid, does initial YOLO inference, using data/temp folder as project/name, then puts txt labels where they need to be and deletes project/name. then passes through human checking, then packages up test and valid as yolo ready with yaml

-training YOLO: give the training set images and labels paths, num epochs, model path. package into YOLO format with yaml then train for those epochs, output path of newest model. use 'model' folder as project

-active loop inference through YOLO: give the unlabelled images folder and model path, return labels in right location, maybe use data/temp project/run folder to store then delete. include conf in txt files - can we get a dictionary in memory so we dont have to search every txt file? investigate results object

-train set updater: scans recently labelled txt files, moves high confidence into train set, samples from set of low confidence, puts into temp folder and asks for human annotation. moves human annotation into train set. if save_all_iterations, then creates new iter_1,2,x folder which saves new versions of training set and unlabelled pool. these versions would only have the frame.txt files, and it would just be a record of at that particular iteration, which frames were in the train set (and their annotations). saves us from having to keep all the images, we can just keep a historical record of what human+pseudo detections had been okayed up to that point .make data/active_loop_datasets so that dont need to bother with data/custom/frames or data/preprocessed/frames
 
-model evaluater: takes model path, and Test or Valid argument, and returns metrics, maybe deposits in txt file with iteration number. we can use metrics = model.val() straight after model.train(), then access metrics.box.map . this would use model.val() with a data=data.yaml that contains the TEST set in place of the val set inside. currently no yolov8 option to get test evaluation, so need to manually switch


