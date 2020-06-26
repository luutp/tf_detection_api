#!/bin/bash
# DEFINES
#!/bin/bash
# DEFINES
WORKSPACE_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd )"
# Get project name from path (basename)
PROJECT_NAME="${PWD##*/}"
PIP_REQUIREMENTS=$WORKSPACE_FOLDER/requirements.txt
CONDA_ENV=tf1
INSTALLATION=false
CONFIG_FILEPATH=$WORKSPACE_FOLDER/config.json
# ======================================================================================
if [ $INSTALLATION == true ]; then
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
