#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
PROJECT_NAME="${PWD##*/}"
CONDA_ENV=tf1
DATASET_NAME='raccoon'
IMAGE_DIR=$WORKSPACE_FOLDER/images
ANNO_DIR=$WORKSPACE_FOLDER/annotations
# Activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
conda env list
cd $WORKSPACE_FOLDER
# Custom function
curr_dir=$PWD
function makedir(){
    local dir="$1"
    if [ -d ${dir} ]; then
        echo "Directory ${dir} exists. Skip mkdir"
        return
    fi
    echo "Makedir: ${dir}"
    cd $(dirname ${dir})
    mkdir $(basename ${dir})
    cd $curr_dir
}
# Run python module
makedir data
python make_tfrecord.py --image_path=$IMAGE_DIR --anno_path=$ANNO_DIR --name=$DATASET_NAME