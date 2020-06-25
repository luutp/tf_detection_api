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
import shutil
import sys

# FileIO Packages
import csv
import json

# Utilities
import logging


# =====================================MAIN==========================================


def makedir(inputDir, remove=False):
    """Summary:
    --------
    Make directory

    Inputs:
    -------
    inputDir (str): fullpath to directory to be created
    remove (bool): option to remove current existing folder
    """
    if remove is True and os.path.exists(inputDir):
        logging.warning("Remove existing folder")
        shutil.rmtree(inputDir)

    if not os.path.exists(inputDir):
        logging.info("Making directory: {}".format(os.path.abspath(inputDir)))
        os.makedirs(inputDir)
    else:
        logging.info(
            "mkdir: Directory already exist: {}".format(os.path.abspath(inputDir))
        )


def load_json(json_filepath):
    output = None
    if not os.path.isfile(json_filepath):
        logging.warning(f"{json_filepath} is not a file")
        return output
    with open(json_filepath, "r") as fid:
        output = json.load(fid)
    return output


def save_json(json_data, json_filepath):
    with open(json_filepath, "w") as fid:
        json.dump(json_data, fid)


def download_tar(url, output_dir):
    with requests.get(url,stream = True) as File:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with open(tmp_file.name,'wb') as fd:
                for chunk in File.iter_content(chunk_size=128):
                    fd.write(chunk)
    
    
def main(**kwargs):
    pass


# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
