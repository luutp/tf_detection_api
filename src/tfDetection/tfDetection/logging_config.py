#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logging_util.py
Description:

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2020/06/11
"""

# =====================================IMPORT PACKAGES==================================
from __future__ import print_function

# Standard Packages
import sys

# Utilities
import logging
from datetime import datetime


# DEFINES
ymd = datetime.now().strftime("%y%m%d")
# ====================================START=============================================
class colorFormatter(logging.Formatter):
    """
    Summary:
    --------
    Customize color for logging

    Inputs:
    -------
        logging (logging): logging
    """

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    Bold = " \x1b[1m"
    Cyan = " \x1b[38;5;122m"
    Magenta = " \x1b[38;5;200m"
    reset = "\x1b[0m"
    msgformat = (
        "{bold}%(asctime)s|%(filename)s:%(lineno)d|%(levelname)s"
        "|{reset}{magenta} %(message)s".format(bold=Bold, reset=reset, magenta=Magenta)
    )

    FORMATS = {
        logging.DEBUG: grey + msgformat + reset,
        logging.INFO: Cyan + msgformat + reset,
        logging.WARNING: yellow + msgformat + reset,
        logging.ERROR: red + msgformat + reset,
        logging.CRITICAL: bold_red + msgformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%y%m%d-%I:%M")
        return formatter.format(record)


def logging_setup():
    """
    Summary:
    --------
    Setup logging for custom format

    Returns:
    --------
        logging: custom logging
    """
    stream_hdl = logging.StreamHandler(sys.stdout)
    stream_hdl.setFormatter(colorFormatter())
    logger = logging.getLogger()
    logger.addHandler(stream_hdl)
    logger.setLevel(logging.INFO)
    # Only keep one logger
    for h in logger.handlers[:-1]:
        logger.removeHandler(h)
    return logger


logger = logging_setup()

# =====================================MAIN=============================================
def main(**kwargs):
    logger.info("hihi")


# =====================================DEBUG============================================
if __name__ == "__main__":
    main()
