#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tfrecord.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/23
"""

# ================================IMPORT PACKAGES====================================
#%%
import argparse

# Standard Packages
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd
from IPython.display import display


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
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


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
    return parser


def main():
    # args = get_args_parser().parse_args()
    # dataset_name = args.name
    # image_path = args.image_path
    # anno_path = args.anno_path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(project_dir,"images")
    anno_path = os.path.join(project_dir,"annotations")
    xml_df = xml_to_df(anno_path)
    display(xml_df)
    # output_dir = os.path.join(os.path.dirname(image_path), "data")
    # output_csv_filepath = os.path.join(output_dir, f"{dataset_name}_labels.csv")
    # xml_df.to_csv(output_csv_filepath, index=None)
    # print("Successfully converted xml to csv.")


main()
#%%
# =====================================DEBUG=========================================

if __name__ == "__main__":
    main()
