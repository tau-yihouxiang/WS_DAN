# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Bird data to TFRecords of TF-Example protos.

This module downloads the Bird data, uncompresses it, reads the files
that make up the Bird data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import numpy as np

import tensorflow as tf

from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0
# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    Bird_root = os.path.join(dataset_dir, 'images')
    directories = []
    class_names = []
    for filename in os.listdir(Bird_root):
        path = os.path.join(Bird_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'Bird_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    if not os.path.exists(os.path.join(dataset_dir, 'tfrecords')):
        os.makedirs(os.path.join(dataset_dir, 'tfrecords'))
    return os.path.join(dataset_dir, 'tfrecords', output_filename)


def _convert_dataset(split_name, dataset, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'testing'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(dataset) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()
        
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        with tf.Session(config=config) as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(dataset))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting %s image %d/%d shard %d' % (split_name,
                            i+1, len(dataset), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(dataset[i]['filename'], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        label = dataset[i]['label']

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, label)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def generate_datasets(data_root):
    train_test = np.loadtxt(os.path.join(data_root, 'train_test_split.txt'), int)
    images_files = np.loadtxt(os.path.join(data_root, 'images.txt'), str)
    labels = np.loadtxt(os.path.join(data_root, 'image_class_labels.txt'), int) - 1
    # parts = np.loadtxt(os.path.join(data_root, 'parts',  'part_locs.txt'), float)
    # parts = np.reshape(parts, [-1, 15, parts.shape[-1]])
    #
    # bboxes = np.loadtxt(os.path.join(data_root, 'bounding_boxes.txt'), float)

    train_dataset = []
    test_dataset = []

    # train_index = 0
    # eval_index = 0
    for index in range(len(images_files)):
        images_file = images_files[index, 1]
        is_training = train_test[index, 1]
        label = labels[index, 1]
        # part = np.reshape(parts[index, :, 2:4], [-1]).tolist()
        # exist = parts[index, :, 4].astype(np.int64).tolist()
        # bbox = bboxes[index, 1:].tolist()

        example = {}
        example['filename'] = os.path.join(data_root, 'images', images_file)
        example['label'] = label
        # example['part'] = part
        # example['exist'] = exist
        # example['bbox'] = bbox

        if is_training:
            train_dataset.append(example)
        else:
            test_dataset.append(example)

    return train_dataset, test_dataset


def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Divide into train and test:
    random.seed(_RANDOM_SEED)

    train_dataset, test_dataset = generate_datasets(dataset_dir)

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # First, convert the training and testing sets.
    _convert_dataset('train', train_dataset, dataset_dir)
    _convert_dataset('test', test_dataset, dataset_dir)

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Bird dataset!')
