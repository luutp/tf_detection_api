<h1 align = 'center'> 
Implement Tensorflow Object Detection API on Custom Dataset
</h1>
<p align="center">
<img alt="Ubuntu16" src="https://img.shields.io/badge/Ubuntu-16.04-2d0922">
<img alt="Anaconda3" src="https://img.shields.io/badge/Anaconda-3-blue">
<img alt="Python37" src="https://img.shields.io/badge/python-3.7-blue">
<img alt="tf15 required" src="https://img.shields.io/badge/TensorFlow%20Requirement-1.15-brightgreen">
<img alt="tf20 not supported" src="https://img.shields.io/badge/tf2_Not_Supported-x-red">

<br>
<img alt="GitHub Workflow Status (branch)" src="https://img.shields.io/github/workflow/status/trieuphatluu/tf_detection_api/Python package/master">
<a href="https://codecov.io/gh/trieuphatluu/tf_detection_api">
  <img src="https://codecov.io/gh/trieuphatluu/tf_detection_api/branch/master/graph/badge.svg" />
</a>
<img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/trieuphatluu/tf_detection_api">
<img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed-raw/trieuphatluu/tf_detection_api">

</p>

---


<p align="center">
<img src=".github/images/906d17abcd.jpg" width=80% alt="cat and dog">
</p>

[Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) provides a collection of detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset,etc. These models can be useful for out-of-the-box inference or initializing your models when training on custom datasets. 

Typically, the following steps are required to train on a custom dataset:
<ol> 
    <li> Installation.
    <ul> 
        <li> Anaconda python=3.7 (optional) </li>
        <li> tensorflow 1.15</li>
        <li> Required python packages (i.e., requirements.txt)</li>
        <li> Clone <a href=https://github.com/tensorflow/models> Tensorflow model garden</a> </li>
        <li> Download and install protobuf</li>
    </ul> 
    </li>
    <li>  Prepare custom dataset
    <ul>
        <li>Prepares images (.jpg format)</li>
        <li>Make annotations (.xml format) for each image</li>
        <li>Combine annotation .xml files into a .csv file for train and test set</li>
        <li>Generate tf records from such datasets</li>
    </ul>
    </li>
    <li>  Download <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>pre-trained models </a></li>
    <li> Configure training pipeline (e.g., edit pipeline.config file)</li>
    <li> Prepare options and run ```python object_detection/model_main.py``` </li>
</ol> 

This repos aims to process all of the required steps mentioned above in <span style='background:#ADD8E6; font-weight:bold'>one bash command</span>

## :heavy_check_mark: Requirements: 
<ul> 
    <li> Anaconda3 </li>
    <li> Python 3.7 </li>
</ul> 

## :zap: Usage

``` 
bash main.sh --config_file={PATH_TO_CONFIG_FILE}
```

### :question: Help
``` 
main.sh [--install] [--config_file str] [--env str]

Implement tensorflow (tf-gpu 1.15.0) Object Detection API on custom Dataset.

Requirements: Anaconda3, python 3.7

Args:
    -h|--help           Show this help message
    --install           Install conda env, pip requirements, and tf models API.
    --config_file  str  Path to config file.
                        Default: /tf_detection_api/config.json
    --env          str  Conda environment name.
                        Default: tf1
```

## :page_facing_up: Config file
The contents of the config .json file is as belows:
``` 
{
    "image_dir": "PATH_TO_IMAGES_DIRECTORY",
    "anno_dir": "PATH_TO_ANNOTATIONS_DIRECTORY",
    "id":[1,2],
    "name":["cat","dog"],
    "pretrained_model": "ssd_mobilenet_v1_coco",
    "train_ratio":0.75
}
```
:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> image_dir </span>: absolute path to the local directory that contains all of the images (both train and test sets). If you plan to download images from google search, Selenium could be a good choice to automate this process. Detailed isntructions can be found in [[4]](#4)

:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> anno_dir </span>: absolute path to annotations directory that contains .xml annotaiton files. labelImg is a nice tool to generate the annotation .xml file from input images. Details can be found in [[5]](#5)

:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> id </span>: list of category IDs from your custom dataset. :memo: Note that the id starts from 1 because id=0 is used for background as default.

:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> name </span>: list of category names from custom dataset

:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> pretrained_model </span>: pre-trained model name from tensorflow model garden. [Full list of pretrained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
<p align="center">
<img src=".github/images/model_zoo.png" width=80% alt="model zoo">
</p>

:black_circle: <span style='color: red; border-style:solid; border-color:#dadfe1;font-style: italic'> train_ratio </span> Ratio to split train and test sets. Default: 0.75


## :memo: More Details

### Project Files (Original)

``` 
.
├── images
│   ├── 0006fee860.jpg
│   ├── 002bcc5167.jpg
│   ├── 0058af494a.jpg
│   ├── ...
├── annotations
│   ├── 0006fee860.xml
│   ├── 002bcc5167.xml
│   ├── 0058af494a.xml
│   └── ...
├── config.json
├── requirements.txt
├── Scripts
│   ├── main.sh
│   ├── predict.sh
├── src
│   └── tfDetection
│       ├── setup.py
│       ├── tfDetection
│       │   ├── __init__.py
│       │   ├── logging_config.py
│       │   ├── predict.py
│       │   ├── train_setup.py
│       │   └── utils.py

```

### Installation
If the option --install in the bash file main.sh is True, the following steps for enviromental setup will be executed
<ol> 
    <li> Create conda virtual environment, python3.7. Default name: tf1 </li>
    <li> Install tensorlfow-gpu==1.15</li>
    <li> Install required python packages in requirements.txt </li>
    <li> Clone and install object detection API from
    <p> https://github.com/tensorflow/models.git  </p> </li>
    <li> Download and install protobuf from 
    <p> https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip </p> </li>
    <li> Install object detection packages</li>
    
</ol> 

### Train setup

#### Prepare dataset
<ol> 
    <li> label map  </li>
    <li> train/test csv file </li>
    <li> train/test tfrecord files </li>
</ol> 

![Prepare dataset](.github/images/dataset.png)


#### Configure training pipeline
![Configure training Pipeline](.github/images/training.png)

#### Automatically generate train.sh bash file 

``` 
PIPELINE_CONFIG_PATH={AUTO} 
MODEL_DIR={AUTO}
NUM_TRAIN_STEPS=10000 
SAMPLE_1_OF_N_EVAL_EXAMPLES=1 
python object_detection/model_main.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--model_dir=${MODEL_DIR} \
	--num_train_steps=${NUM_TRAIN_STEPS} \
	--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
	--alsologtostderr \

```

#### Train model on custom dataset

![ckpts](.github/images/ckpts.png)

## <a id=ref >:clipboard: References </a>
<a id="1">[1]</a> 
Tensorflow model garden installation
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

<a id="2">[2]</a> 
Setup for custom dataset
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

<a id="3">[3]</a> 
Run the traning job
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md

<a id="4">[4]</a> 
Search and Download image from google with Python and Selenium
https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d

<a id="5">[5]</a> 
Label images with labelImg tool
https://github.com/tzutalin/labelImg

## Licence
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">