import numpy as np
import os
import re

import tensorflow as tf

import tf_nn_util

from AlphaGo.preprocessing.preprocessing import Preprocess, DEFAULT_FEATURES
from AlphaGo.util import flatten_idx


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


class CNNPolicy:
    """uses a convolutional neural network to evaluate the state of the game
    and compute a probability distribution over the next action
    """

    def __init__(self,
                 feature_list=DEFAULT_FEATURES,
                 bsize=19,
                 filters=192,
                 checkpoint_dir='./logs'):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        self.bsize = bsize
        self.preprocessor = Preprocess(feature_list)
        self.rows = bsize
        self.cols = bsize
        self.input_depth = self.preprocessor.output_dim
        self.filters = filters

        self.checkpoint_dir = checkpoint_dir

        self.sess = None
        self.probs = None
        self.statesholder = None

    def init_graph(self, weight_setter=None, train=False, learning_rate=1e-03):
        # initialize computation graph
        self.g = tf.Graph()	
        with self.g.as_default():
            self.statesholder = self._statesholder()
            self.actionsholder = self._actionsholder()
            self.rewardsholder = self._rewardsholder()

            self.probs = self.inference(self.statesholder, weight_setter)

            if train:
                self.accuracy_op = self.accuracy(self.probs, self.actionsholder)
                self.loss_op = self.loss(self.probs, self.actionsholder, self.rewardsholder)
                self.train_op = self.train(self.loss_op, learning_rate)

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def start_session(self, config):
        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init_op)

    def close_session(self):
        self.sess.close()

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_model(self, global_step=None):
        if global_step is not None:
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)
        else:
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'))

    def eval_state(self, state, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.forward(tensor)
        moves = moves or state.get_legal_moves()
        return self._select_moves_and_normalize(network_output[0], moves, state.size)

    def batch_eval_state(self, states, moves_lists=None):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.

        Analogous to [eval_state(s) for s in states]

        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []
        state_size = states[0].size
        if not all([st.size == state_size for st in states]):
            raise ValueError("all states must have the same size")
        # conshcatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)
        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        network_output = self.forward(nn_input)
        # default move lists to all legal moves
        moves_lists = moves_lists or [st.get_legal_moves() for st in states]
        results = [None] * n_states
        for i in range(n_states):
            results[i] = self._select_moves_and_normalize(network_output[i], moves_lists[i],
                                                          state_size)
        return results

    def forward(self, states):
        states = self.reordering_states_tensor(states)
        return self.sess.run(self.probs,
                             feed_dict={self.statesholder: states})

    def run_train(self, step, states, actions, rewards):
        states = self.reordering_states_tensor(states)
        feed_dict = {
            self.statesholder: states,
            self.actionsholder: actions,
            self.rewardsholder: rewards
        }

        loss, accuracy, _ = self.sess.run([self.loss_op, self.accuracy_op, self.train_op],
                                          feed_dict=feed_dict)

        return loss, accuracy

    def reordering_states_tensor(self, states):
        # TF dim ordering is different to default keras input implementation
        return states.transpose((0, 2, 3, 1))

    def _statesholder(self):
        return tf.placeholder(tf.float32,
                              shape=(None,
                                     self.bsize,
                                     self.bsize,
                                     self.input_depth))

    def _actionsholder(self):
        return tf.placeholder(tf.float32,
                              shape=(None, self.bsize**2))

    def _rewardsholder(self):
        return tf.placeholder(tf.float32, shape=(None,))

    def inference(self, statesholder, weight_setter=None):

        def initial_weight(layer, wb, scope_name):
            if wb.lower() == 'w':
                if layer == 1:
                    return tf_nn_util.variable_with_weight_decay(
                                scope_name + '_W',
                                [5, 5, self.input_depth, self.filters])
                elif layer <= 12:
                    return tf_nn_util.variable_with_weight_decay(
                                scope_name + '_W',
                                [3, 3, self.filters, self.filters])
                elif layer == 13:
                    return tf_nn_util.variable_with_weight_decay(
                                scope_name + '_W',
                                [1, 1, self.filters, 1])
            elif wb.lower() == 'b':
                if 1 <= layer and layer <= 12:
                    return tf_nn_util.zero_variable(
                                scope_name + '_b',
                                [self.filters])
                elif layer == 13:
                    return tf_nn_util.zero_variable(
                                scope_name + '_b',
                                [1])
                elif layer == 14:
                    return tf_nn_util.zero_variable(
                                'Variable',
                                [self.bsize**2])

        if not weight_setter:
            weight_setter = initial_weight

        # convolution2d_1
        with tf.variable_scope('convolution2d_1') as scope:
            weights = weight_setter(1, 'w', scope.name)
            biases = weight_setter(1, 'b', scope.name)
            conv = tf.nn.conv2d(statesholder, weights, [1, 1, 1, 1], padding='SAME')
            bias_add = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias_add, name=scope.name)

        # convolution2d_2-12
        convi = conv1
        for i in range(2, 13):
            with tf.variable_scope('convolution2d_' + str(i)) as scope:
                weights = weight_setter(i, 'w', scope.name)
                biases = weight_setter(i, 'b', scope.name)
                conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
                bias_add = tf.nn.bias_add(conv, biases)
                conv = tf.nn.relu(bias_add, name=scope.name)
            convi = conv

        # convolution2d_13
        with tf.variable_scope('convolution2d_13') as scope:
            weights = weight_setter(13, 'w', scope.name)
            biases = weight_setter(13, 'b', scope.name)
            conv = tf.nn.conv2d(convi, weights, [1, 1, 1, 1], padding='SAME')
            conv13 = tf.nn.bias_add(conv, biases, name=scope.name)

        # linear
        with tf.variable_scope('bias_1') as scope:
            bias = weight_setter(14, 'b', scope.name)
            flatten = tf.reshape(conv13, [-1, self.bsize**2])
            linear = tf.add(flatten, bias, name=scope.name)

        # softmax
        with tf.variable_scope('softmax') as scope:
            logits = tf.nn.softmax(linear)

        return logits

    def loss(self, probs, actionsholder, rewardsholder):
        with tf.variable_scope('loss') as scope:
            clip_probs = tf.clip_by_value(probs, 1e-07, 1.0)
            good_probs = tf.reduce_sum(tf.mul(clip_probs, actionsholder), reduction_indices=[1])

            loss = tf.neg(tf.reduce_mean(tf.log(good_probs)), name=scope.name)
            # eligibility = tf.mul(tf.log(good_probs), rewardsholder)
            # loss = tf.neg(tf.reduce_mean(eligibility), name=scope.name)
        return loss

    def accuracy(self, probs, actionsholder):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(probs, tf.argmax(actionsholder, 1), 1)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32), name=scope.name)
        return acc

    def train(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return train_op

    def activation_summary(x):
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

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        return zip(moves, distribution)
