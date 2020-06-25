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
import xml.etree.ElementTree as ET
from collections import namedtuple
from collections import OrderedDict

# FileIO Packages
import io
import json
import requests

# Data Analytics
import numpy as np
import pandas as pd

# DL Frameworks
import tensorflow as tf

# Utilities
import logging

# Custom Packages
from IPython.display import display
from lxml import html
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from PIL import Image
from tfDetection import utils


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
    print(f"Making {output_dir}/train_labels.csv")
    df_train.to_csv(train_csvpath, index=None)
    print(f"Making {output_dir}/test_labels.csv")
    df_test.to_csv(test_csvpath, index=None)


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
        print(f"Successfully created the TFRecords: {output_path}")


def download_pretrained_model(cfg):
    url = cfg["model_zoo_url"]
    page = requests.get(url)
    webpage = html.fromstring(page.content)
    results = webpage.xpath("//a/@href")
    results = list(filter(lambda x: "download.tensorflow.org" in x, results))
    pretrained_url = None
    for r in results:
        if cfg["pretrain_model"] in r:
            pretrained_url = r
            break
    if pretrained_url is not None:
        

list_pretrained_model(None)
#%%


def get_args_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(os.path.abspath(__file__)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Additional Info",
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="full path to cfg file"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.75, help="Train samples ratio"
    )
    return parser


def main(args):
    # config_filepath = args.config_filepath
    # train_ratio = args.train_ratio
    config_filepath = os.path.join(
        os.path.expanduser("~"), "tf_detection_api/config.json"
    )
    train_ratio = 0.75
    cfg = dict()
    with open(config_filepath, "r") as fid:
        cfg = json.load(fid)
    cfg["project_dir"] = os.path.dirname(cfg["image_dir"])
    cfg["dataset_dir"] = os.path.join(cfg["project_dir"], "datasets")
    cfg["label_map"] = os.path.join(cfg["dataset_dir"], "label_map.pbtxt")
    cfg["train_csvpath"] = os.path.join(cfg["dataset_dir"], "train_labels.csv")
    cfg["test_csvpath"] = os.path.join(cfg["dataset_dir"], "test_labels.csv")
    cfg["train_record"] = os.path.join(cfg["dataset_dir"], "train.record")
    cfg["test_record"] = os.path.join(cfg["dataset_dir"], "test.record")
    cfg["config_filepath"] = config_filepath
    cfg["train_ratio"] = train_ratio
    cfg[
        "model_zoo_url"
    ] = "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md"

    make_label_map(cfg)
    make_traintest_csv(cfg)
    make_tfrecord(cfg)


# main(None)
#%%
# =====================================DEBUG=========================================

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
