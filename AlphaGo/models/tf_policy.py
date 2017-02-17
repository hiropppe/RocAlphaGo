import numpy as np
import os

import tensorflow as tf

import tf_nn_util

from AlphaGo.preprocessing.preprocessing import Preprocess, DEFAULT_FEATURES
from AlphaGo.util import flatten_idx


class CNNPolicy:
    """uses a convolutional neural network to evaluate the state of the game
    and compute a probability distribution over the next action
    """

    def __init__(self,
                 feature_list=DEFAULT_FEATURES,
                 bsize=19,
                 filters=192,
                 checkpoint_dir='./logs',
                 weight_setter=None):
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

        # initialize computation graph
        with tf.Graph().as_default():
            self.statesholder = self._statesholder()

            self.logits = self.inference(weight_setter)

            self.saver = tf.train.Saver()

            self.sess = tf.Session()

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_model(self):
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

    def _statesholder(self):
        return tf.placeholder(tf.float32,
                              shape=(None,
                                     self.bsize,
                                     self.bsize,
                                     self.input_depth))

    def forward(self, states):
        # dim ordering is different to keras input
        states = states.transpose((0, 2, 3, 1))
        return self.sess.run(self.logits,
                             feed_dict={self.statesholder: states})

    def inference(self, weight_setter=None):

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
            conv = tf.nn.conv2d(self.statesholder, weights, [1, 1, 1, 1], padding='SAME')
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
