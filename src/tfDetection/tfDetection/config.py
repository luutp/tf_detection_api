#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/25
"""
#%%
# ================================IMPORT PACKAGES====================================

# Standard Packages
import os

# FileIO Packages
import json

# Custom Packages
import tfDetection.utils as utils


config_filepath = os.path.join(os.path.expanduser("~"), "tf_detection_api/config.json")
config = dict()
with open(config_filepath, "r") as fid:
    config = json.load(fid)
config["project_dir"] = os.path.dirname(config["image_dir"])
config["dataset_dir"] = os.path.join(config["project_dir"], "datasets")
config["label_map"] = "label_map.pbtxt"
config["config_filepath"] = config_filepath

# =====================================MAIN==========================================

def main(**kwargs):
    pass


# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
