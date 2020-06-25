#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
PROJECT_NAME="${PWD##*/}"
CONDA_ENV=tf1
DATASET_NAME='datasets'
IMAGE_DIR=$WORKSPACE_FOLDER/images
ANNO_DIR=$WORKSPACE_FOLDER/annotations
TRAIN_RATIO=0.75
# Activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
conda env list
cd $WORKSPACE_FOLDER/src
# Run python module
python train_setup.py --config_filepath=$CONFIG_FILEPATH --train_ratio=$TRAIN_RATIO