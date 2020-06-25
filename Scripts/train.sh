#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
PROJECT_NAME="${PWD##*/}"
CONDA_ENV=tf1
MODEL_NAME="ssd_mobilenet_v1_coco_2018_01_28"
MODEL_URL="http://download.tensorflow.org/models/object_detection/$MODEL_NAME.tar.gz"
# Activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
conda env list

cd $HOME/gitClone/models/research
echo $PWD
echo "Add PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd $HOME/gitClone/models/research

if [ "$1" == "" ]; then
    echo "Training Model"
    PIPELINE_CONFIG_PATH=$WORKSPACE_FOLDER/training/pipeline.config
    MODEL_DIR=$WORKSPACE_FOLDER/ckpts
    NUM_TRAIN_STEPS=50000
    SAMPLE_1_OF_N_EVAL_EXAMPLES=1
    python object_detection/model_main.py \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --model_dir=${MODEL_DIR} \
        --num_train_steps=${NUM_TRAIN_STEPS} \
        --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
        --alsologtostderr
fi

if [ "$1" == "export" ]; then
    echo "Exporting model"
    INPUT_TYPE=image_tensor
    PIPELINE_CONFIG_PATH="$WORKSPACE_FOLDER/training/pipeline.config"
    TRAINED_CKPT_PREFIX="$WORKSPACE_FOLDER/ckpts/model.ckpt-496"
    EXPORT_DIR="$WORKSPACE_FOLDER/ckpts/deploy"
    mkdir $EXPORT_DIR
    python object_detection/export_inference_graph.py \
        --input_type=${INPUT_TYPE} \
        --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
        --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
        --output_directory=${EXPORT_DIR}
fi
