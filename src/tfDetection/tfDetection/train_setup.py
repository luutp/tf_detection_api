#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tfrecord.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/23
"""
#%%

# ================================IMPORT PACKAGES=====================================

# Standard Packages
import argparse
import glob
import os
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from collections import namedtuple

# FileIO Packages
import io
import json
import requests

# Data Analytics
import numpy as np
import pandas as pd

# DL Frameworks
import tensorflow as tf

# Custom Packages
from lxml import html
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from PIL import Image
from pyfiglet import Figlet
from tfDetection import utils
from tfDetection.logging_config import logger as logging
from tfDetection.utils import printit


# =====================================MAIN=============================================


def xml_to_df(path):
    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    df = pd.DataFrame(xml_list, columns=column_name)
    return df


@printit
def make_traintest_csv(cfg):
    df = xml_to_df(cfg["anno_dir"])
    train_ratio = cfg["train_ratio"]
    output_dir = cfg["dataset_dir"]
    train_csvpath = cfg["train_csvpath"]
    test_csvpath = cfg["test_csvpath"]

    gb = df.groupby("filename")
    grouped_list = [gb.get_group(x) for x in gb.groups]
    nb_samples = len(grouped_list)
    train_index = np.random.choice(
        nb_samples, size=int(train_ratio * nb_samples), replace=False
    )
    test_index = np.setdiff1d(list(range(nb_samples)), train_index)
    df_train = pd.concat([grouped_list[i] for i in train_index])
    df_test = pd.concat([grouped_list[i] for i in test_index])
    if output_dir is None:
        logging.error("output_dir is not defined")
        return
    utils.makedir(output_dir)
    logging.info(f"Making {output_dir}/train_labels.csv")
    df_train.to_csv(train_csvpath, index=None)
    logging.info(f"Making {output_dir}/test_labels.csv")
    df_test.to_csv(test_csvpath, index=None)


@printit
def make_label_map(cfg):
    id_list = cfg["id"]  # id starts with 1
    name_list = cfg["name"]
    output_dir = cfg["dataset_dir"]
    utils.makedir(output_dir)
    output_filepath = cfg["label_map"]
    with open(output_filepath, "w") as fid:
        for id, name in zip(id_list, name_list):
            fid.write("item { \n")
            fid.write(f"\tid: {id}\n")
            fid.write(f"\tname: '{name}'\n")
            fid.write("}\n")


def class_text_to_int(label_map_filepath, input_label):
    label_dict = label_map_util.get_label_map_dict(label_map_filepath)
    for key, val in label_dict.items():
        if key == input_label:
            return val


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(label_map_filepath, group, path):
    with tf.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(label_map_filepath, row["class"]))

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


@printit
def make_tfrecord(cfg):
    image_dir = cfg["image_dir"]
    for output_path, csv_input in zip(
        [cfg["train_record"], cfg["test_record"]],
        [cfg["train_csvpath"], cfg["test_csvpath"]],
    ):
        writer = tf.python_io.TFRecordWriter(output_path)
        path = os.path.join(image_dir)
        examples = pd.read_csv(csv_input)
        grouped = split(examples, "filename")
        for group in grouped:
            tf_example = create_tf_example(cfg["label_map"], group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        logging.info(f"Successfully created the TFRecords: {output_path}")


@printit
def download_pretrained_model(cfg):
    url = cfg["model_zoo_url"]
    pretrained_model = cfg["pretrained_model"]
    page = requests.get(url)
    webpage = html.fromstring(page.content)
    url_list = webpage.xpath("//a/@href")
    url_list = list(
        filter(lambda x: "download.tensorflow.org" and "tar.gz" in x, url_list)
    )
    pretrained_url = None
    for url in url_list:
        if pretrained_model in url:
            pretrained_url = url
            break
    if pretrained_url is not None:
        utils.download_url(pretrained_url, cfg["pretrained_filepath"])
        model_tarfile = os.path.basename(cfg["pretrained_filepath"])
        logging.info(f"Extracting {model_tarfile}")
        with tarfile.open(cfg["pretrained_filepath"]) as fid:
            fid.extractall(cfg["training_dir"])
        logging.info(f"Removing {model_tarfile}")
        os.remove(cfg["pretrained_filepath"])
    else:
        logging.info(f"{pretrained_url} is not available for downloading")


@printit
def make_pipeline_config_file(cfg):
    root_dir = cfg["training_dir"]
    ext = ".config"
    config_filepath = ""
    for path, subdirs, files in os.walk(root_dir):
        for filename in files:
            name, this_ext = os.path.splitext(filename)
            if this_ext == ext:
                config_filepath = os.path.join(path, filename)
    model_ckpt = os.path.join(os.path.dirname(config_filepath), "model.ckpt")
    file_contents = []
    with open(config_filepath, "r") as fid:
        for line in fid:
            if "model.ckpt" in line:
                line = f'\tfine_tune_checkpoint: "{model_ckpt}"\n'
            if "label_map.pbtxt" in line:
                line = f'\tlabel_map_path: "{cfg["label_map"]}"\n'
            if "train.record" in line:
                line = f'\t\tinput_path: "{cfg["train_record"]}"\n'
            if "val.record" in line:
                line = f'\t\tinput_path: "{cfg["test_record"]}"\n'

            file_contents.append(line)
    with open(cfg["pipeline_config_filepath"], "w") as fid:
        for line in file_contents:
            fid.write(line)
    logging.info(f"{cfg['pipeline_config_filepath']} has been created")


@printit
def make_train_bash_file(cfg):
    with open(cfg["train_bash_filepath"], "w") as fid:
        fid.write(f"PIPELINE_CONFIG_PATH={cfg['pipeline_config_filepath']} \n")
        fid.write(f"MODEL_DIR={cfg['ckpts_dir']} \n")
        fid.write(f"NUM_TRAIN_STEPS=10000 \n")
        fid.write(f"SAMPLE_1_OF_N_EVAL_EXAMPLES=1 \n")
        fid.write(f"python object_detection/model_main.py \\\n")
        fid.write("\t--pipeline_config_path=${PIPELINE_CONFIG_PATH} \\\n")
        fid.write("\t--model_dir=${MODEL_DIR} \\\n")
        fid.write("\t--num_train_steps=${NUM_TRAIN_STEPS} \\\n")
        fid.write("\t--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \\\n")
        fid.write("\t--alsologtostderr \\\n")


# =====================================MAIN=============================================
def get_args_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(os.path.abspath(__file__)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Additional Info",
    )
    parser.add_argument(
        "--config_filepath", type=str, required=True, help="full path to cfg file"
    )
    return parser


def main(args):
    # config_filepath = os.path.join(
    #     os.path.expanduser("~"), "tf_detection_api/config.json"
    # )
    # train_ratio = 0.75
    config_filepath = args.config_filepath
    cfg = dict()
    with open(config_filepath, "r") as fid:
        cfg = json.load(fid)
    cfg["project_dir"] = os.path.dirname(cfg["image_dir"])
    cfg["dataset_dir"] = os.path.join(cfg["project_dir"], "datasets")
    cfg["training_dir"] = os.path.join(cfg["project_dir"], "training")
    cfg["label_map"] = os.path.join(cfg["dataset_dir"], "label_map.pbtxt")
    cfg["train_csvpath"] = os.path.join(cfg["dataset_dir"], "train_labels.csv")
    cfg["test_csvpath"] = os.path.join(cfg["dataset_dir"], "test_labels.csv")
    cfg["train_record"] = os.path.join(cfg["dataset_dir"], "train.record")
    cfg["test_record"] = os.path.join(cfg["dataset_dir"], "test.record")
    cfg["config_filepath"] = config_filepath
    cfg[
        "model_zoo_url"
    ] = "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md"
    cfg["pretrained_filepath"] = os.path.join(
        cfg["training_dir"], f"{cfg['pretrained_model']}.tar.gz"
    )
    cfg["pipeline_config_filepath"] = os.path.join(
        cfg["training_dir"], "training_pipeline.config"
    )
    cfg["train_bash_filepath"] = os.path.join(cfg["project_dir"], "Scripts/train.sh")
    cfg["ckpts_dir"] = os.path.join(cfg["project_dir"], "ckpts")

    make_label_map(cfg)
    make_traintest_csv(cfg)
    make_tfrecord(cfg)
    download_pretrained_model(cfg)
    make_pipeline_config_file(cfg)
    make_train_bash_file(cfg)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    f = Figlet(font="slant")
    print(f.renderText("TF - Detection"))
    # main(args)
