#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
PROJECT_NAME="${PWD##*/}"
CONDA_ENV=tf1
# Run train_setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
conda env list
cd $HOME/gitClone/models/research
echo "Add PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

echo "Exporting model"
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH="$WORKSPACE_FOLDER/training/training_pipeline.config"
TRAINED_CKPT_PREFIX="$WORKSPACE_FOLDER/ckpts/model.ckpt-4460"
EXPORT_DIR="$WORKSPACE_FOLDER/ckpts/deploy"
mkdir $EXPORT_DIR
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}