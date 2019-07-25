# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import time

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './Car/Sample/TRAIN/bilinear_center_loss_attention_crop_soft_drop',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', './Car/Sample/EVAL/bilinear_center_loss_attention_crop_soft_drop',
    'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'car', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', './Car/Data',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 448, 'Eval image size')

tf.app.flags.DEFINE_string(
    'gpus', "3",
    'gpu devices')

tf.app.flags.DEFINE_string(
    'feature_maps', 'Mixed_6e',
    'the layer name of feature maps')

tf.app.flags.DEFINE_string(
    'attention_maps', 'Mixed_7a_b0',
    'the layer name of attention maps')

tf.app.flags.DEFINE_integer(
    'num_parts', None,
    'number of parts'
)

FLAGS = tf.app.flags.FLAGS


def add_eval_summary(logits, labels, scope=''):
    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)
    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval%s/%s' % (scope, name)
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    return names_to_updates


def draw_keypints(image, keypoints, exist):
    for i in range(exist.size):
        if exist[i] == 1:
            cv2.circle(image, center=(int(keypoints[2 * i]), int(keypoints[2 * i + 1])), radius=5, color=(255, 0, 0),
                       thickness=-1)

    return image


import numpy as np
import cv2, os
import random, shutil


def visualization(images, feature_maps, logits):
    index_dir = str(random.randint(0, 100))
    visual_dir = os.path.join('./CUB-200-2011/visualization', index_dir)

    if os.path.exists(visual_dir):
        shutil.rmtree(visual_dir)
    os.makedirs(visual_dir)

    img = ((images[0] + 1) * 127).astype(np.uint8)
    # img = (images[0] + 128).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(visual_dir, 'image.jpg'), img)

    feature_map = feature_maps[0]
    mean_feature = np.mean(feature_map, axis=-1, keepdims=True)
    mean_feature = (mean_feature / np.max(mean_feature) * 255).astype(np.uint8)
    mean_feature = cv2.resize(mean_feature, (100, 100))
    cv2.imwrite(os.path.join(visual_dir, 'mean_feature.jpg'), mean_feature)

    max_feature = np.max(feature_map, axis=-1, keepdims=True)
    max_feature = (max_feature / np.max(max_feature) * 255).astype(np.uint8)
    max_feature = cv2.resize(max_feature, (100, 100))
    cv2.imwrite(os.path.join(visual_dir, 'max_feature.jpg'), max_feature)

    feature_map = (feature_map / np.max(feature_map) * 255).astype(np.uint8)
    for index in range(feature_maps.shape[-1]):
        feature = np.expand_dims(feature_map[:, :, index], axis=2)
        feature = cv2.resize(feature, (100, 100))
        cv2.imwrite(os.path.join(visual_dir, '%s.jpg' % index), feature)

    return logits


def predict_results(images, feature_maps, logits, labels):
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        logit = logits[i]

        index_dir = str(np.argmax(logit))
        visual_dir = os.path.join('./CUB-200-2011/predict_results', index_dir)
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

        img = ((image + 1) * 127).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_name = str(label) + '_' + str(random.randint(1, 10000)) + '.jpg'

        feature_map = feature_maps[i]
        mean_feature = np.mean(feature_map, axis=-1, keepdims=True)
        mean_feature = (mean_feature / np.max(mean_feature, keepdims=True) * 255).astype(np.uint8)
        mean_feature = cv2.resize(mean_feature, (image.shape[0], image.shape[1]))
        mean_feature = np.reshape(mean_feature, [image.shape[0], image.shape[1], 1])

        mean_feature = np.tile(mean_feature, [1, 1, 3])

        showImg = np.concatenate([img, mean_feature], axis=1)
        cv2.imwrite(os.path.join(visual_dir, image_name), showImg)

    return logits


def mask2bbox(attention_maps):
    height = attention_maps.shape[1]
    width = attention_maps.shape[2]
    bboxes = []
    for i in range(attention_maps.shape[0]):
        mask = attention_maps[i]
        max_activate = mask.max()
        min_activate = 0.1 * max_activate
        mask = (mask >= min_activate)
        itemindex = np.where(mask == True)

        ymin = itemindex[0].min() / height - 0.05
        ymax = itemindex[0].max() / height + 0.05
        xmin = itemindex[1].min() / width - 0.05
        xmax = itemindex[1].max() / width + 0.05

        bbox = np.asarray([ymin, xmin, ymax, xmax], dtype=np.float32)
        bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=5 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################

        logits_1, end_points_1 = network_fn(images)

        attention_maps = tf.reduce_mean(end_points_1['attention_maps'], axis=-1, keepdims=True)
        attention_maps = tf.image.resize_bilinear(attention_maps, [eval_image_size, eval_image_size])
        bboxes = tf.py_func(mask2bbox, [attention_maps], [tf.float32])
        bboxes = tf.reshape(bboxes, [FLAGS.batch_size, 4])
        box_ind = tf.range(FLAGS.batch_size, dtype=tf.int32)

        images = tf.image.crop_and_resize(images, bboxes, box_ind, crop_size=[eval_image_size, eval_image_size])
        logits_2, end_points_2 = network_fn(images, reuse=True)

        logits = tf.log(tf.nn.softmax(logits_1) * 0.5 + tf.nn.softmax(logits_2) * 0.5)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        logits_to_updates = add_eval_summary(logits, labels, scope='/bilinear')
        logits_1_to_updates = add_eval_summary(logits_1, labels, scope='/logits_1')
        logits_2_to_updates = add_eval_summary(logits_2, labels, scope='/logits_2')

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.visible_device_list = FLAGS.gpus

        while True:
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path

            tf.logging.info('Evaluating %s' % checkpoint_path)

            eval_op = list(logits_to_updates.values())
            eval_op.extend(list(logits_1_to_updates.values()))
            eval_op.extend(list(logits_2_to_updates.values()))

            slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=eval_op,
                variables_to_restore=variables_to_restore,
                session_config=config)

            time.sleep(60 * 5)
            # break


if __name__ == '__main__':
    tf.app.run()
