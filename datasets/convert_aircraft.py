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
r"""Downloads and converts cub data to TFRecords of TF-Example protos.

This module downloads the cub data, uncompresses it, reads the files
that make up the cub data and creates two TFRecord datasets: one for train
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
    images_root = os.path.join(dataset_dir, 'images')
    directories = []
    class_names = []
    for filename in os.listdir(images_root):
        path = os.path.join(images_root, filename)
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
    output_filename = 'Aircraft_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    if not os.path.exists(os.path.join(dataset_dir, 'tfrecords')):
        os.makedirs(os.path.join(dataset_dir, 'tfrecords'))
    return os.path.join(dataset_dir, 'tfrecords', output_filename)


def _convert_dataset(split_name, datasets, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(datasets) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list='1'
        with tf.Session(config=config) as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(datasets))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(datasets), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(datasets[i]['filename'], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_id = datasets[i]['label']

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def generate_datasets(data_root):
    train_info = np.loadtxt(os.path.join(data_root, 'fgvc-aircraft-2013b/data', 'images_variant_trainval.txt'), str)
    test_info = np.loadtxt(os.path.join(data_root, 'fgvc-aircraft-2013b/data', 'images_variant_test.txt'), str)
    category_info = np.loadtxt(os.path.join(data_root, 'fgvc-aircraft-2013b/data', 'variants.txt'), str)

    train_dataset = []
    test_dataset = []
    for index in range(len(train_info)):
        images_file = os.path.join(data_root, 'fgvc-aircraft-2013b/data/images', train_info[index, 0] + '.jpg')
        category = train_info[index, 1]
        label = np.where(category_info == category)[0][0]

        example = {}
        example['filename'] = images_file
        example['label'] = int(label)
        train_dataset.append(example)
        
    for index in range(len(test_info)):
        images_file = os.path.join(data_root, 'fgvc-aircraft-2013b/data/images', test_info[index, 0] + '.jpg')
        category = test_info[index, 1]
        label = np.where(category_info == category)[0][0]

        example = {}
        example['filename'] = images_file
        example['label'] = int(label)
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

    # First, convert the training and test sets.
    _convert_dataset('train', train_dataset, dataset_dir)
    _convert_dataset('test', test_dataset, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the fgvc dataset!')
