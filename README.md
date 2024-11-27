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