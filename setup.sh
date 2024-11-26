#!/bin/bash

# ----------------------------------------------------------------
# Setup file for object detection pipeline
# > Creates a virtual environment to handle library dependencies
# ----------------------------------------------------------------

# Setup a Python virtual environment 
python3 -m venv obj_det_venv
# Activate the virtual environment
source obj_det_venv/bin/activate
# Install library dependencies listed in requirements.txt file
pip install -r requirements.txt

echo "Environment setup complete!"