#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/25
"""

# ================================IMPORT PACKAGES====================================

# Standard Packages
import os

# Custom Packages
import tfDetection.utils as utils


config = dict()
config["image_dir"] = os.path.join(os.path.expanduser("~"), "tf_detection_api/images")
config["anno_dir"] = os.path.join(
    os.path.expanduser("~"), "tf_detection_api/annotations"
)
config["project_dir"] = os.path.dirname(config["image_dir"])
config["dataset_dir"] = os.path.join(config["project_dir"], "datasets")
config["label_map"] = "label_map.pbtxt"
config["config_file"] = "config.json"

# =====================================MAIN==========================================


def make_label_map(output_dir):
    id_list = [1]  # id starts with 1
    name_list = ["raccoon"]
    utils.makedir(output_dir)
    output_filepath = os.path.join(output_dir, config["label_map"])
    with open(output_filepath, "w") as fid:
        for id, name in zip(id_list, name_list):
            fid.write("item { \n")
            fid.write(f"\tid: {id}\n")
            fid.write(f"\tname: {name}\n")
            fid.write("}\n")


def main(**kwargs):
    # make_label_map(output_dir=config["dataset_dir"])
    utils.save_json(config, os.path.join(config['project_dir'], 'config.json'))


# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
