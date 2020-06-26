#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
PROJECT_NAME="${PWD##*/}"
PIP_REQUIREMENTS=$WORKSPACE_FOLDER/requirements.txt
CONDA_ENV=tf1
INSTALLATION=false
CONFIG_FILEPATH=$WORKSPACE_FOLDER/config.json

# Display help message
USAGE="$(basename "$0") [--install] [--config_file str] [--env str]

Implement tensorflow (tf-gpu 1.15.0) Object Detection API on custom Dataset.

Requirements: Anaconda3, python 3.7

Args:
    -h|--help           Show this help message
    --install           Install conda env, pip requirements, and tf models API.
    --config_file  str  Path to config file.
                        Default: $CONFIG_FILEPATH
    --env          str  Conda environment name.
                        Default: $CONDA_ENV
    "

# Arguments parser
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) show_help=true ;;
        --install) INSTALLATION=true ;;
        --config_file) CONFIG_FILEPATH="$2"; shift ;;
        --env) CONDA_ENV="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

if [[ $show_help == true ]]; then
    echo "$USAGE"
    exit
fi

# ======================================================================================
if [[ $INSTALLATION == true ]]; then
    echo "Installing tensorflow object detection API"
    # Create Conda environment
    if [ ! -d $HOME/anaconda3/envs/$CONDA_ENV  ]; then
        echo "Setup Conda $CONDA_ENV Environment"
        conda create -y -n $CONDA_ENV python=3.7
    fi
    # Activate
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
    conda env list
    conda install tensorflow-gpu==1.15
    # pip install
    echo "Installing PIP requirements"
    /home/$USER/anaconda3/envs/$CONDA_ENV/bin/pip install -r $PIP_REQUIREMENTS
    # Download tensorflow models api
    if [ ! -d $HOME/gitClone  ]; then
        echo "Making gitClone Directory"
        mkdir $HOME/gitClone
    fi
    echo "cd to $HOME/gitClone"
    cd $HOME/gitClone
    echo "Cloning from https://github.com/tensorflow/models.git"
    git clone https://github.com/tensorflow/models.git
    echo "cd to $HOME/gitClone/models/research"
    cd $HOME/gitClone/models/research
    echo "Protobuf compilation"
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
    unzip protobuf.zip
    ./bin/protoc object_detection/protos/*.proto --python_out=.
    echo "pip install object_detection package"
    pip install .
    echo "Add PYTHONPATH"
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    echo "Verify Installation"
    python object_detection/builders/model_builder_tf1_test.py
fi
# Run train_setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
conda env list
cd $HOME/gitClone/models/research
echo "Add PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# Run python module
python $WORKSPACE_FOLDER/src/tfDetection/tfDetection/train_setup.py --config_filepath=$CONFIG_FILEPATH
# Run train.sh
bash $WORKSPACE_FOLDER/Scripts/train.sh
