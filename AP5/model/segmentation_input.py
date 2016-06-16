# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:50:47 2015.

@author: teichman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def _load_gt_file(hypes, data_file=None):
    """Take the data_file and hypes and create a generator.

    The generator outputs the image and the gt_image.
    """
    base_path = os.path.realpath(os.path.dirname(data_file))
    files = [line.rstrip() for line in open(data_file)]

    for epoche in itertools.count():
        shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), \
                "File does not exist: %s" % gt_image_file
            image = scipy.misc.imread(image_file)
            gt_image = scp.misc.imread(gt_image_file, flatten=True)

            yield image, gt_image


def _make_data_gen(hypes, phase, data_dir):
    """Return a data generator that outputs image samples."""
    if phase == 'train':
        data_file = hypes['data']["train_file"]
    elif phase == 'val':
        data_file = hypes['data']["val_file"]
    else:
        assert False, "Unknown Phase %s" % phase

    data_file = os.path.join(data_dir, data_file)

    fg_color = np.array(hypes['data']['instrument_color'])
    background_color = np.array(hypes['data']['background_color'])

    data = _load_gt_file(hypes, data_file)

    for image, gt_image in data:

        gt_bool = (gt_image != background_color)

        if phase == 'val':
            assert image.shape[:-1] == gt_image.shape, \
                ("image.shape: {0},"
                 "gt_image.shape: {1}").format(image.shape[:-1],
                                               gt_bool.shape)
            yield image, gt_bool
        elif phase == 'train':

            yield jitter_input(hypes, image, gt_bool)

            yield jitter_input(hypes, np.fliplr(image), np.fliplr(gt_bool))

            yield jitter_input(hypes, np.flipud(image), np.flipud(gt_bool))

            yield jitter_input(hypes, np.flipud(np.fliplr(image)),
                               np.flipud(np.fliplr(gt_bool)))


def jitter_input(hypes, image, gt_image):
    lower_size = 0.4
    upper_size = 1.7
    sig = 0.25
    res_chance = 0.5

    max_crop = 32
    crop_chance = 0.95

    if res_chance > random.random():
        image, gt_image = random_resize(image, gt_image, lower_size,
                                        upper_size, sig)
        image, gt_image = crop_to_size(hypes, image, gt_image)

    if crop_chance > random.random():
        image, gt_image = random_crop(image, gt_image, max_crop)

    assert image.shape[:-1] == gt_image.shape, \
        "image.shape: {0}, gt_image.shape: {1}".format(image.shape[:-1],
                                                       gt_image.shape)

    assert(image.shape[:-1] == gt_image.shape)
    return image, gt_image


def random_crop(image, gt_image, max_crop):
    offset_x = random.randint(1, max_crop)
    offset_y = random.randint(1, max_crop)

    if random.random() > 0.5:
        image = image[offset_x:, offset_y:]
        gt_image = gt_image[offset_x:, offset_y:]
    else:
        image = image[:-offset_x, :-offset_y]
        gt_image = gt_image[:-offset_x, :-offset_y]

    return image, gt_image


def random_resize(image, gt_image, lower_size, upper_size, sig):
    factor = random.normalvariate(1, sig)
    if factor < lower_size:
        factor = lower_size
    if factor > upper_size:
        factor = upper_size
    image = scipy.misc.imresize(image, factor)
    gt_image = scipy.misc.imresize(gt_image, factor, interp='nearest')
    gt_image = gt_image/255
    return image, gt_image


def crop_to_size(hypes, image, gt_image):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = hypes['arch']['image_width']
    height = hypes['arch']['image_height']
    if new_width > width:
        max_x = max(new_height-height, 0)
        max_y = new_width-width
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)
        image = image[offset_x:offset_x+height, offset_y:offset_y+width]
        gt_image = gt_image[offset_x:offset_x+height, offset_y:offset_y+width]

    return image, gt_image


def create_queues(hypes, phase):
    """Create Queues."""
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]
    shapes = (
        [arch['image_height'], arch['image_width'], arch['num_channels']],
        [arch['image_height'], arch['image_width'], arch['num_classes']],)
    capacity = 50
    q = tf.FIFOQueue(capacity=50, dtypes=dtypes)
    tf.scalar_summary("queue/%s/fraction_of_%d_full" %
                      (q.name + "_" + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def start_enqueuing_threads(hypes, q, phase, sess, data_dir):
    """Start enqueuing threads."""
    shape = [hypes['arch']['image_height'], hypes['arch']['image_width'],
             hypes['arch']['num_channels']]
    image_pl = tf.placeholder(tf.float32)

    # Labels
    shape = [hypes['arch']['image_height'], hypes['arch']['image_width'],
             hypes['arch']['num_classes']]
    label_pl = tf.placeholder(tf.int32)

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):
        # infinity loop enqueueing data
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    threads = []
    enqueue_op = q.enqueue((image_pl, label_pl))
    gen = _make_data_gen(hypes, phase, data_dir)
    gen.next()
    # sess.run(enqueue_op, feed_dict=make_feed(data))
    if phase == 'val':
        num_threads = 1
    else:
        num_threads = hypes["solver"]["threads"]
    for i in range(num_threads):
        threads.append(tf.train.threading.Thread(target=enqueue_loop,
                                                 args=(sess, enqueue_op,
                                                       phase, gen)))
    threads[-1].start()


def _read_processed_image(hypes, q, phase):
    image, label = q.dequeue()
    if phase == 'train':
        image
        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        image = tf.image.random_brightness(image, max_delta=35)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        # image = tf.image.random_hue(image, max_delta=0.05)
        # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    return image, label


def _dtypes(tensor_list_list):
    all_types = [[t.dtype for t in tl] for tl in tensor_list_list]
    types = all_types[0]
    for other_types in all_types[1:]:
        if other_types != types:
            raise TypeError("Expected types to be consistent: %s vs. %s." %
                            (", ".join(x.name for x in types),
                             ", ".join(x.name for x in other_types)))
    return types


def _enqueue_join(queue, tensor_list_list):
    enqueue_ops = [queue.enqueue(tl) for tl in tensor_list_list]
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))


def shuffle_join(tensor_list_list, capacity,
                 min_ad, phase):
    name = 'shuffel_input'
    types = _dtypes(tensor_list_list)
    queue = data_flow_ops.RandomShuffleQueue(
        capacity=capacity, min_after_dequeue=min_ad,
        dtypes=types)

    # Build enque Operations
    _enqueue_join(queue, tensor_list_list)

    full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_ad),
                          dtypes.float32) * (1. / (capacity - min_ad)))
    # Note that name contains a '/' at the end so we intentionally do not place
    # a '/' after %s below.
    summary_name = (
        "queue/%s/fraction_over_%d_of_%d_full" %
        (name + '_' + phase, min_ad, capacity - min_ad))
    tf.scalar_summary(summary_name, full)

    dequeued = queue.dequeue(name='shuffel_deqeue')
    # dequeued = _deserialize_sparse_tensors(dequeued, sparse_info)
    return dequeued


def inputs(hypes, q, phase, data_dir):
    """Generate Inputs images."""
    if phase == 'val':
        image, label = _read_processed_image(hypes, q, phase)
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        return image, label

    num_threads = hypes["solver"]["threads"]
    example_list = [_read_processed_image(hypes, q, phase)
                    for i in range(num_threads)]
    minad = 50
    capacity = minad + 5
    image, label = shuffle_join(example_list, capacity, minad, phase)
    tensor_name = image.op.name
    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)

    nc = hypes["arch"]["num_classes"]
    label.set_shape([1, None, None, 1])

    # Display the training images in the visualizer.
    tf.image_summary(tensor_name + '/image', image)

    label = tf.expand_dims(label, 3)
    tf.image_summary(tensor_name + '/gt_image', tf.to_float(label))

    return image, label


def main():
    """main."""
    with open('../hypes/medseg.json', 'r') as f:
        hypes = json.load(f)

    q = {}
    q['train'] = create_queues(hypes, 'train')
    q['val'] = create_queues(hypes, 'val')
    data_dir = os.environ['TV_DIR_DATA']

    _make_data_gen(hypes, 'train', data_dir)

    image_batch, label_batch = inputs(hypes, q['train'], 'train', data_dir)

    logging.info("Start running")

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        start_enqueuing_threads(hypes, q['train'], 'train', sess, data_dir)

        logging.info("Start running")
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in itertools.count():
            image, gt = sess.run([image_batch, label_batch])
            # scp.misc.imshow(image[0])
            # scp.misc.imshow(gt[0])

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
