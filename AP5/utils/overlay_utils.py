"""Utils for Overlay in Semantic Segmentation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import json
import logging
import numpy as np
import os
import sys
import time

from PIL import Image
import scipy.ndimage

from datetime import datetime

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import tensorflow as tf
import matplotlib.cm as cm


def make_soft_overlay(image, gt_prob):
    """
    Overlays image with propability map.
    Overlays the image with a colormap ranging
    from blue to red according to the probability map
    given in gt_prob. This is good to analyse the segmentation
    result of a single class.
    Parameters
    ----------
    image : np array
        an image of shape [width, height, 3]
    gt_prob : np array
        propability map for one class
        with shape [width, height]
    return: overlay
        a blue vs. red overlay of the propability map
    """

    mycm = cm.get_cmap('bwr')

    overimage = mycm(gt_prob, bytes=True)
    output = 0.4*overimage[:, :, 0:3] + 0.6*image

    return output


def overlay_segmentation(image, segmentation, color_dict):
    """
    Overlays the image with a hard segmentation result.
    The function can be applied to display the (hard)
    segmentation result of arbitrary many classes.
    Parameters
    ----------
    image : np array
        an image of shape [width, height, 3]
    segmentation : numpy array
        segmentation map of shape [width, height]
    color_changes : dict
        The key is the class and the value is the Color the class is drawn
        in the overlay.
        Each color has to be a tuple (r, g, b, a) with r, g, b, a in
        {0, 1, ..., 255}.
        Choose a = 0 for (invisible) background and a = 127 is recommended.
    Returns
    -------
    np.array
        The new colored segmentation
    """
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_dict:
                output.putpixel((y, x), color_dict[segmentation[x, y]])
            elif 'default' in color_dict:
                output.putpixel((y, x), color_dict['default'])

    background = scipy.misc.toimage(image)
    background.paste(output, box=None, mask=output)

    return np.array(background)
