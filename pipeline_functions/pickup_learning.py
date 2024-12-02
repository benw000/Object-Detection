import torch
from ultralytics import YOLO

'''
Script to quickly train and compare different YOLO models.
Adjust this to point to your own dataset, and play about with model arguments.

I use this to further train an imported pretrained model (YOLOv11 on COCO dataset).
With default arguments, if you train again it seems to lose all its knowledge.
Solutions may be to reduce learning rate so new changes are less destructive,
and also to freeze YOLO's NN layers up to a certain point, using their 'freeze' argument.

I refine the pretrained model on my own labelled data,
then use both the base model and refined model to predict on a set of unlabelled images.
I use rough metrics like number of object detections to assert whether refined model is worse.
If refined model picks up 0 detections while base model picks up 200, then refined model is worse.
'''

device = torch.device("mps")
num_epochs = 5

dataset = "pickup_learning/data/dataset.yaml"
classes=[32]

freeze_up_to = None # 9 is the end of YOLO's backbone feature extraction layers

# Learning rates
small = 1*(10**-8)
medium = 5*(10**-5)
big = 1*(10**-4)
large = 1*(10**-3)
init_learning_rate = small
final_learning_rate = 0.01 # fraction of initial learning rate
warmup_bias_lr = small

optimizer = "AdamW"

# Create base model instance and train/refine it on custom dataset
base_model_path = "pickup_learning/yolo11n.pt"
model = YOLO(base_model_path)
model.to(device)
results = model.train(data=dataset, 
                        project="pickup_learning/runs", name=f"train_{freeze_up_to}_{init_learning_rate}", exist_ok=True,
                        lr0=init_learning_rate, lrf=final_learning_rate, warmup_bias_lr=warmup_bias_lr,
                        optimizer=optimizer,
                        freeze=freeze_up_to,
                        classes=classes, max_det=1, resume=False, verbose=False,
                        plots=True,
                        epochs=num_epochs)
print("Model finished training with mean average precision:")
print(results.box.map)

# Inference on unlabelled set
unlabelled_pool_path = "pickup_learning/data/images/unlabelled"

# Infer with refined model
results = model.predict(source=unlabelled_pool_path,
                                max_det=1,
                                classes=classes,
                                save=True,
                                save_txt=True, save_conf=True,
                                verbose=True,
                                project="pickup_learning/runs", name=f"infer_{freeze_up_to}_{init_learning_rate}",
                                exist_ok = True)


'''
# Set up new instance of base model
model = YOLO(base_model_path)
model.to(device)
# Infer with base model
results = model.predict(source=unlabelled_pool_path,
                                max_det=1,
                                classes=classes,
                                save=True,
                                save_txt=True, save_conf=True,
                                verbose=True,
                                project="pickup_learning/runs", name="base_infer",
                                exist_ok = True)
'''
# Base model expecting around 250 labels
# Results should be found in printout plus various folders
