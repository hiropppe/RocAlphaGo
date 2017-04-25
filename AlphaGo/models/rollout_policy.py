import tensorflow as tf

import tf_nn_util


class RolloutPolicy:

    def __init__(self,
                 n_feature,
                 bsize=19,
                 logdir='./logs'):
        self.bsize = bsize
        self.logdir = logdir
        self.n_feature = n_feature

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.statesholder = tf.placeholder(tf.float32,
                                               shape=(None,
                                                      self.bsize**2,
                                                      self.input_depth))
            self.actionsholder = tf.placeholder(tf.float32,
                                                shape=(None, self.bsize**2))

            self.W = tf_nn_util.variable_with_weight_decay(
                        'W',
                        [self.input_depth])
            self.b = tf_nn_util.zero_variable(
                        'b',
                        [self.bsize**2])

    def inference(self, stateholder):
        with tf.variable_scope('logits') as scope:
            logits = tf.nn.biad_add(tf.nn.matmul(stateholder, self.W), self.b, name=scope.name)
        return logits

    def softmax(self, logits):
        with tf.variable_scope('softmax') as scope:
            probs = tf.nn.softmax(logits, name=scope.name)
        return probs

    def loss(self, logits, actionsholder):
        with tf.variable_scope('loss') as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                  logits, actionsholder), name=scope.name)
        return loss

    def accuracy(self, probs, actionsholder):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(probs, tf.argmax(actionsholder, 1), 1)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32), name=scope.name)
        return acc

    def train(self, loss, learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op
