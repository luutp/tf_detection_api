#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tfrecord.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/23
"""


# Standard Packages
# ================================IMPORT PACKAGES=====================================
#%%
import argparse
import glob
import os
import xml.etree.ElementTree as ET
from collections import namedtuple
from collections import OrderedDict

# FileIO Packages
import io

# Data Analytics
import numpy as np
import pandas as pd

# DL Frameworks
import tensorflow as tf

# Utilities
import logging

# Custom Packages
from IPython.display import display
from object_detection.utils import dataset_util
from PIL import Image
from tfDetection.config import config
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


def make_traintest_csv(df, train_ratio=0.75, output_dir=None):
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
    df_train.to_csv(os.path.join(output_dir, "train_labels.csv"), index=None)
    print(f"Making {output_dir}/test_labels.csv")
    df_test.to_csv(os.path.join(output_dir, "test_labels.csv"), index=None)


def class_text_to_int(row_label):
    if row_label == "raccoon":
        return 1
    else:
        None


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(group, path):
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
        classes.append(class_text_to_int(row["class"]))

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


def get_args_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(os.path.abspath(__file__)),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Additional Info",
    )
    parser.add_argument("--name", type=str, default="datasets", help="Dataset name")
    parser.add_argument(
        "--image_path", type=str, required=False, help="Path to images directory"
    )
    parser.add_argument(
        "--anno_path", type=str, required=False, help="Path to annotations directory"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.75, help="Train samples ratio"
    )
    return parser


def main():
    args = get_args_parser().parse_args()
    dataset_name = args.name
    image_path = args.image_path
    anno_path = args.anno_path
    train_ratio = args.train_ratio

    project_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(project_dir, "images")
    anno_path = os.path.join(project_dir, "annotations")
    output_dir = os.path.join(os.path.dirname(image_path), dataset_name)

    df = xml_to_df(anno_path)
    make_traintest_csv(df, train_ratio=train_ratio, output_dir=output_dir)


#%%
# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
