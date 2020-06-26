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
import functools
import os
import shutil

# FileIO Packages
import json
import requests

# Utilities
from tqdm import tqdm

# Custom Packages
from tfDetection.logging_config import logger as logging


# =====================================MAIN==========================================


def printit(method):
    @functools.wraps(method)
    def inner(*args, **kwargs):
        func_name = method.__name__
        logging.info(f"START: {func_name}.")
        result = method(*args, **kwargs)
        logging.info(f"DONE: {func_name}")
        return result

    return inner


@printit
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


@printit
def load_json(json_filepath):
    output = None
    if not os.path.isfile(json_filepath):
        logging.warning(f"{json_filepath} is not a file")
        return output
    with open(json_filepath, "r") as fid:
        output = json.load(fid)
    return output


@printit
def save_json(json_data, json_filepath):
    with open(json_filepath, "w") as fid:
        json.dump(json_data, fid)


def download_url(url, to_file, **kwargs):
    if os.path.exists(to_file):
        logging.info("File exists: {}. Skip downloading".format(to_file))
        return
    logging.info("Downloading to: {}".format(to_file))
    makedir(os.path.dirname(to_file))
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(to_file, "wb") as fid:
        for data in r.iter_content(block_size):
            t.update(len(data))
            fid.write(data)
    t.close()
    logging.info("\n")


def main(**kwargs):
    pass


# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
