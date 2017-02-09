#!/usr/bin/env python

import os
import re

import numpy as np
import h5py as h5
import tensorflow as tf

from collections import defaultdict

# constant
BOARD_SIZE = 19
NUM_CHANNELS = 48
FILTER_SIZE_1 = 5
FILTER_SIZE_2 = 3

FLAGS = tf.app.flags.FLAGS

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def inference(states_placeholder, initial_weights):
    def get_param(i, wb):
        layer = initial_weights['convolution2d_' + str(i)]
        value = layer['convolution2d_' + str(i) + '_' + wb + ':0'].value
        if wb.upper() == 'W':
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            return value.transpose((2, 3, 1, 0))
        else:
            return value

    # conv1
    with tf.variable_scope('convolution2d_1') as scope:
        weights = tf.Variable(get_param(1, 'W'), name='convolution2d_1_W')
        biases = tf.Variable(get_param(1, 'b'), name='convolution2d_1_b')
        conv = tf.nn.conv2d(states_placeholder, weights, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # conv2-12
    convX = conv1
    for i in range(2, 13):
        with tf.variable_scope('convolution2d_' + str(i)) as scope:
            weights = tf.Variable(get_param(i, 'W'), name='convolution2d_' + str(i) + '_W')
            biases = tf.Variable(get_param(i, 'b'), name='convolution2d_' + str(i) + '_b')
            conv = tf.nn.conv2d(convX, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv)
        convX = conv

    # conv13
    with tf.variable_scope('convolution2d_13') as scope:
        weights = tf.Variable(get_param(13, 'W'), name='convolution2d_13_W')
        biases = tf.Variable(get_param(13, 'b'), name='convolution2d_13_b')
        conv = tf.nn.conv2d(convX, weights, [1, 1, 1, 1], padding='SAME')
        conv13 = tf.nn.bias_add(conv, biases)
        _activation_summary(conv13)

    # linear
    with tf.variable_scope('bias_1') as scope:
        flatten = tf.reshape(conv13, [-1, BOARD_SIZE**2])
        bias = tf.Variable(initial_weights['bias_1']['Variable:0'].value, name='Variable')
        linear = tf.add(flatten, bias, name=scope.name)
        _activation_summary(linear)

    # softmax
    with tf.variable_scope('softmax') as scope:
        probs = tf.nn.softmax(linear)
        _activation_summary(probs)

    return probs


def loss(probs, actions_placeholder, rewards_placeholder):
    with tf.variable_scope('loss') as scope:
        clip_probs = tf.clip_by_value(probs, 1e-07, 1.0)
        good_probs = tf.reduce_sum(tf.mul(clip_probs, actions_placeholder), reduction_indices=[1])

        loss = tf.neg(tf.reduce_mean(tf.log(good_probs)), name=scope.name)
        # eligibility = tf.mul(tf.log(good_probs), rewards_placeholder)
        # loss = tf.neg(tf.reduce_mean(eligibility), name=scope.name)

        _activation_summary(loss)
    return loss


def accuracy(probs, actions_placeholder):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(probs, tf.argmax(actions_placeholder, 1), 1)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32), name=scope.name)
        _activation_summary(acc)
    return acc


def train(loss, learning_rate, rewards_placeholder):
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grads = optimizer.compute_gradients(loss)
    mean_reward = tf.reduce_mean(rewards_placeholder)
    for i, (grad, var) in enumerate(grads):
        if grad is not None:
            grads[i] = (tf.mul(grad, mean_reward), var)
    train_op = optimizer.apply_gradients(grads)
    # train_op = optimizer.minimize(loss)
    return train_op


def save_keras_weights(sess, step):
    """
    Save policy weights
        convolution2d_X/convolution2d_X_W:0
        convolution2d_X/convolution2d_X_b:0
        bias_1/Variable:0
    """
    weights_name = 'weights.{}.hdf5'.format(str(step).rjust(5, '0'))
    with h5.File(os.path.join(FLAGS.train_directory, weights_name), 'w') as root:
        model_weights = root.create_group('model_weights')
        layer_names = []
        weight_names = defaultdict(list)

        for v in tf.trainable_variables()[:27]:
            (layer_name, weight_name) = v.name.split('/')

            if layer_name not in layer_names:
                layer_names.append(layer_name.encode('utf8'))
            weight_names[layer_name].append(weight_name.encode('utf8'))

            weight = sess.run(v.read_value())
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            if re.match('convolution2d_\d+_W:0', weight_name):
                weight = weight.transpose((3, 2, 0, 1))
            if layer_name not in model_weights:
                layer_group = model_weights.create_group(layer_name)
            layer_group.create_dataset(weight_name, data=weight)

        model_weights.attrs['layer_names'] = np.array(layer_names)
        for layer_name in layer_names:
            model_weights[layer_name].attrs['weight_names'] = np.array(weight_names[layer_name])
        # import pdb; pdb.set_trace()
        # print(weights_name)
    return weights_name


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
