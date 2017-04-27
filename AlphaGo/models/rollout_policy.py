import tensorflow as tf

import tf_nn_util


class RolloutPolicy:

    def __init__(self,
                 n_features,
                 bsize=19,
                 logdir='./logs'):
        self.bsize = bsize
        self.logdir = logdir
        self.n_features = n_features

    def inference(self, states_pl, n_features):
        with tf.variable_scope('logits') as scope:
            kernel = tf_nn_util.variable_with_weight_decay(
                        'weights',
                        shape=[1, 1, n_features, 1])
            bias = tf_nn_util.zero_variable(
                        'bias',
                        [self.bsize**2])
            conv = tf.nn.conv2d(states_pl, kernel, strides=[1, 1, 1, 1], padding='SAME')
            flatten = tf.reshape(conv, [-1, self.bsize**2])
            pre_activation = tf.nn.bias_add(flatten, bias)
            logits = tf.nn.relu(pre_activation, name=scope.name)
        return logits

    def softmax(self, logits):
        with tf.variable_scope('softmax') as scope:
            probs = tf.nn.softmax(logits, name=scope.name)
        return probs

    def loss(self, logits, actions_pl):
        with tf.variable_scope('loss') as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                  logits, actions_pl), name=scope.name)
        return loss

    def accuracy(self, probs, actions_pl):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(probs, tf.argmax(actions_pl, 1), 1)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32), name=scope.name)
        return acc

    def train(self, loss, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
