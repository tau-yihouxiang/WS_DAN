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
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_utils

import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_bap_base(inputs,
                      final_endpoint='Mixed_6e',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    """Inception model from http://arxiv.org/abs/1512.00567.

    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.

    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.

    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256a  | Mixed_5b
    mixed_35x35x288a  | Mixed_5c
    mixed_35x35x288b  | Mixed_5d
    mixed_17x17x768a  | Mixed_6a
    mixed_17x17x768b  | Mixed_6b
    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
        'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      scope: Optional variable_scope.

    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 35 x 35 x 192.

        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                           scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    end_points[end_point + '_b1'] = branch_1
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    end_points[end_point + '_b0'] = branch_0
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    end_points[end_point + '_b0'] = branch_0
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                           scope='Conv2d_0c_7x1')
                    end_points[end_point + '_b1'] = branch_1
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    end_points[end_point + '_b0'] = branch_0
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v3_bap_head(net,end_points,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None,
                      ):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(scope, 'InceptionV3', [net], reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                            padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                            scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                            scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                            padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                    end_points[end_point + '_b0'] = branch_0
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)
    return net, end_points


def attention_pooling(attention_maps, feature_maps):
    attention_features = []
    for attention_map in tf.unstack(attention_maps, axis=-1):
        attention_map = tf.expand_dims(attention_map, 3)
        part_feature = attention_map * feature_maps
        attention_features.append(part_feature)
    attention_features = tf.concat(attention_features, axis=0)
    return attention_features


def bilinear_attention_pooling(feature_maps, attention_maps, end_points, name):
    feature_shape = feature_maps.get_shape().as_list()
    attention_shape = attention_maps.get_shape().as_list()

    phi_I = tf.einsum('ijkm,ijkn->imn', attention_maps, feature_maps)
    phi_I = tf.divide(phi_I, tf.to_float(attention_shape[1] * attention_shape[2]))
    phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))

    raw_features = tf.nn.l2_normalize(phi_I, axis=[1, 2])
    raw_features = tf.reshape(raw_features, [-1, 1, 1, attention_shape[-1] * feature_shape[-1]])
    end_points[name] = raw_features

    pooling_features = raw_features * 100.0
    return pooling_features, end_points

def wsddn_pooling(feature_maps, attention_maps, keep_prob, end_points, name):
    feature_shape = feature_maps.get_shape().as_list()
    attention_shape = attention_maps.get_shape().as_list()

    phi_I = tf.einsum('ijkm,ijkn->imn', attention_maps, feature_maps)
    phi_I = tf.divide(phi_I, tf.to_float(attention_shape[1] * attention_shape[2]))
    phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))

    phi_attention = phi_I / tf.reduce_sum(phi_I + 1e-12, axis=1, keepdims=True)
    phi_feature = phi_I / tf.reduce_sum(phi_I + 1e-12, axis=2, keepdims=True)
    phi_I = phi_attention * phi_feature

    pooling_features = tf.reduce_sum(phi_I, axis=1)
    pooling_features = tf.nn.l2_normalize(pooling_features, axis=-1)
    end_points[name] = tf.reshape(pooling_features, [-1, 1, 1, feature_shape[-1]])

    pooling_features = tf.reshape(pooling_features * 100.0, [-1, 1, 1, feature_shape[-1]])

    return pooling_features, end_points

def aspp_residual(attention_maps):
    scale1 = attention_maps
    scale2 = slim.conv2d(attention_maps, 192, [3, 3], rate=6)
    scale3 = slim.conv2d(attention_maps, 192, [3, 3], rate=12)
    scale4 = slim.conv2d(attention_maps, 192, [3, 3], rate=18)
    attention_maps = scale1 + scale2 + scale3 + scale4
    return attention_maps

def generate_attention_image(image, attention_map):
    h, w, _ = image.shape
    mask = np.mean(attention_map, axis=-1, keepdims=True)
    mask = (mask / np.max(mask) * 255.0).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))

    image = (image / 2.0 + 0.5) * 255.0
    image = image.astype(np.uint8)

    color_map = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    attention_image = cv2.addWeighted(image, 0.5, color_map.astype(np.uint8), 0.5, 0)
    attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
    return attention_image


def inception_v3_bap(inputs,
                 num_classes=1000,
                 is_training=True,
                 min_depth=16,
                 depth_multiplier=1.0,
                 reuse=False,
                 scope='InceptionV3'):

    with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v3_bap_base(
                inputs, final_endpoint='Mixed_7c', scope=scope, min_depth=min_depth,
                depth_multiplier=depth_multiplier)

            with tf.variable_scope('bilinear_attention_pooling'):

                feature_maps = end_points[FLAGS.feature_maps]
                attention_maps = end_points[FLAGS.attention_maps]

                num_parts = FLAGS.num_parts
                attention_maps = attention_maps[:, :, :, :num_parts]

                attention_image = tf.py_func(generate_attention_image, [inputs[0], attention_maps[0]], tf.uint8)
                tf.summary.image('attention_image', tf.expand_dims(attention_image, 0))
                tf.summary.image('input_image', inputs[0:1])
                if is_training:
                    tf.summary.image('attention_maps', tf.reduce_mean(attention_maps[0:1], axis=-1, keepdims=True))
                    tf.summary.image('feature_maps', tf.reduce_mean(feature_maps[0:1], axis=-1, keepdims=True))

                end_points['attention_maps'] = attention_maps
                end_points['feature_maps'] = feature_maps

                bap_features, end_points = bilinear_attention_pooling(feature_maps, attention_maps
                                                                         , end_points, 'embeddings')

                with tf.variable_scope('logits'):
                    logits = slim.conv2d(bap_features, num_classes, [1, 1],
                                              activation_fn=None, normalizer_fn=None)
                    logits = tf.squeeze(logits, [1, 2])
                    end_points['logits'] = logits

        return logits, end_points

inception_v3_bap.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.

    TODO(jrru): Make this function work with unknown shapes. Theoretically, this
    can be done with the code below. Problems are two-fold: (1) If the shape was
    known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
    handle tensors that define the kernel size.
        shape = tf.shape(input_tensor)
        return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                           tf.minimum(shape[2], kernel_size[1])])

    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


inception_v3_bap_arg_scope = inception_utils.inception_arg_scope
