#!/usr/bin/env python

import re

import tensorflow as tf

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
        W = layer['convolution2d_' + str(i) + '_' + wb + ':0'].value
        # keras conv2d shape (192, 48, 5, 5) -> raw tensorflow conv2d shape (5, 5, 48, 192)
        return W.transpose()

    # conv1
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(get_param(1, 'W'), 'convolution2d_1_W')
        biases = tf.Variable(get_param(1, 'b'), 'convolution2d_1_b')
        conv = tf.nn.conv2d(states_placeholder, weights, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # conv2-12
    convX = conv1
    for i in range(2, 13):
        with tf.variable_scope('conv' + str(i)) as scope:
            weights = tf.Variable(get_param(i, 'W'), 'convolution2d_' + str(i) + '_W')
            biases = tf.Variable(get_param(i, 'b'), 'convolution2d_' + str(i) + '_b')
            conv = tf.nn.conv2d(convX, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv)
        convX = conv

    # conv13
    with tf.variable_scope('conv13') as scope:
        weights = tf.Variable(get_param(13, 'W'), 'convolution2d_13_W')
        biases = tf.Variable(get_param(13, 'b'), 'convolution2d_13_b')
        conv = tf.nn.conv2d(convX, weights, [1, 1, 1, 1], padding='SAME')
        conv13 = tf.nn.bias_add(conv, biases)
        _activation_summary(conv13)

    # linear
    with tf.variable_scope('linear') as scope:
        flatten = tf.reshape(conv13, [-1, BOARD_SIZE**2])
        bias = initial_weights['bias_1']['Variable:0'].value
        linear = tf.add(flatten, bias, name=scope.name)
        _activation_summary(linear)

    # softmax
    with tf.variable_scope('softmax') as scope:
        probs = tf.nn.softmax(linear)
        _activation_summary(probs)

    return probs


def loss(probs, actions_placeholder, rewards_placeholder):
    with tf.variable_scope('loss') as scope:
        good_probs = tf.reduce_sum(tf.mul(probs, actions_placeholder), reduction_indices=[1])
        loss = -tf.reduce_mean(tf.log(good_probs), name=scope.name)
        # eligibility = tf.mul(tf.log(good_probs), rewards_placeholder)
        # loss = -tf.reduce_mean(eligibility)
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
