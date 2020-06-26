#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/23
"""
#%%

# ================================IMPORT PACKAGES=====================================

# Standard Packages
import argparse
import os

# FileIO Packages
import json

# Data Analytics
import numpy as np

# DL Frameworks
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

# Visualization Packages
from PIL import Image

# Custom Packages
from IPython.display import display
from tfDetection.logging_config import logger as logging


tf.compat.v1.enable_eager_execution()


# =====================================START============================================


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }
    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)
    # Handle models with masks:
    if "detection_masks" in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"],
            output_dict["detection_boxes"],
            image.shape[0],
            image.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path, category_index):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=4,
    )
    display(Image.fromarray(image_np))


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
    # config_filepath = args.config_filepath
    config_filepath = os.path.join(
        os.path.expanduser("~"), "tf_detection_api/config.json"
    )
    cfg = dict()
    with open(config_filepath, "r") as fid:
        cfg = json.load(fid)
    cfg["project_dir"] = os.path.dirname(cfg["image_dir"])
    cfg["dataset_dir"] = os.path.join(cfg["project_dir"], "datasets")
    cfg["label_map"] = os.path.join(cfg["dataset_dir"], "label_map.pbtxt")
    PATH_TO_LABELS = cfg["label_map"]
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True
    )
    model_dir = os.path.join(cfg["project_dir"], "ckpts/deploy/saved_model")
    model = tf.compat.v2.saved_model.load(str(model_dir))
    model = model.signatures["serving_default"]

    test_image = os.path.join(cfg["image_dir"], "raccoon-130.jpg")

    show_inference(model, test_image, category_index)


main(None)
#%%

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
