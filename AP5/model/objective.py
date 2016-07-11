#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from utils.kitti_devkit import seg_utils as seg
import utils.overlay_utils as ov


import tensorflow as tf


def decoder(hypes, logits):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return:
      logits: the logits are already decoded.
    """
    return logits


def loss(hypes, logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """

    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, 2))
        labels = tf.to_int64(tf.reshape(labels, (-1,)))
        # shape = [logits.get_shape()[0], 2]
        # epsilon = tf.constant(value=hypes['solver']['epsilon'], shape=shape)
        # logits = logits + epsilon

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='xentropy')

        head = 1 + tf.to_float(labels)*(hypes['solver']['head']-1)

        cross_entropy = cross_entropy*head

        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    background_color = np.array(hypes['data']['background_color'])
    gt_bool = (gt_image != background_color)

    FN, FP, posNum, negNum = seg.evalExp(gt_bool, cnn_image,
                                         thresh, validMap=None,
                                         validArea=None)

    return FN, FP, posNum, negNum


def evaluate(hypes, sess, image_pl, softmax):
    """Do evaluate in python.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    sess : tensorflow session
        The session which has the computational graph.
    image_pl: placeholder
        A placeholder for images
    softmax: tensor
        The softmax output.
    Returns
    -------
    eval_list: List of tuples
        Contains numerical evaluation result and name of the metric.
    image_list: List of images
        List of images drawn as overlay.
    """
    data_dir = hypes['dirs']['data_dir']
    data_file = hypes['data']['val_file']
    data_file = os.path.join(data_dir, data_file)
    image_dir = os.path.dirname(data_file)

    thresh = np.array(range(0, 256))/255.0
    total_fp = np.zeros(thresh.shape)
    total_fn = np.zeros(thresh.shape)
    total_posnum = 0
    total_negnum = 0

    image_list = []

    with open(data_file) as file:
        for i, datum in enumerate(file):
                datum = datum.rstrip()
                image_file, gt_file = datum.split(" ")
                image_file = os.path.join(image_dir, image_file)
                gt_file = os.path.join(image_dir, gt_file)

                image = scp.misc.imread(image_file)
                gt_image = scp.misc.imread(gt_file)
                shape = image.shape

                feed_dict = {image_pl: image}

                output = sess.run([softmax], feed_dict=feed_dict)
                output_im = output[0][:, 1].reshape(shape[0], shape[1])

                if True:
                    ov_image = ov.make_soft_overlay(image, 1-output_im)
                    name = os.path.basename(image_file)
                    image_list.append((name, ov_image))

                FN, FP, posNum, negNum = eval_image(hypes, gt_image, output_im)

                total_fp += FP
                total_fn += FN
                total_posnum += posNum
                total_negnum += negNum

    eval_dict = seg.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                            total_fn, total_fp,
                                            thresh=thresh)

    eval_list = []

    eval_list.append(('MaxF1', 100*eval_dict['MaxF']))
    eval_list.append(('BestThresh', 100*eval_dict['BestThresh']))
    eval_list.append(('Average Precision', 100*eval_dict['AvgPrec']))

    accuracy = eval_dict['accuracy']
    eval_list.append(('Max Accuricy', 100*np.max(accuracy)))
    eval_list.append(('Acc. Tresh',
                      100*eval_dict['thresh'][np.argmax(accuracy)]))

    ind5 = np.where(eval_dict['thresh'] >= 0.5)[0][0]
    ind25 = np.where(eval_dict['thresh'] >= 0.25)[0][0]

    eval_list.append(('Accuracy @ 0.5', 100*accuracy[ind5]))
    eval_list.append(('Accuracy @ 0.25', 100*accuracy[ind25]))

    return eval_list, image_list
