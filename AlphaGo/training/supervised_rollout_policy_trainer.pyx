#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
"""
#cython: wraparound=False
"""
import numpy as np

cimport numpy as np

import os
import tensorflow as tf

from tqdm import tqdm

from AlphaGo import go
from AlphaGo.util import sgf_iter_states
from AlphaGo.preprocessing import rollout_preprocessing
from AlphaGo.models import rollout_policy

flags = tf.app.flags
flags.DEFINE_integer("board_size", 19, "")

flags.DEFINE_string("sgf_dir", "", "")
flags.DEFINE_string('logdir', '/tmp/logs',
                    'Directory where to save latest parameters.')

flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')

flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_boolean('verbose', True, '')
FLAGS = flags.FLAGS




class SizeMismatchError(Exception):
    pass


def _is_sgf(fname):
    return fname.strip()[-4:] == ".sgf"


def _list_sgfs(path):
    """helper function to get all SGF files in a directory (does not recurse)
    """
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if _is_sgf(f))


def run_training(policy):
    cdef np.ndarray[DTYPE_t, ndim=2] F

    sgf_files = _list_sgfs(FLAGS.sgf_dir)
    for sgf_file in tqdm(sgf_files):
        with open(sgf_file, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)
        
            for (state, move, player) in state_action_iterator:
                if state.size != FLAGS.bd_size:
                    raise SizeMismatchError()
                if move != go.PASS_MOVE:
                    nn_input = self.feature_processor.state_to_tensor(state)


def main(argv=None):
    # start a server for a specific task
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                            device_count={'GPU': 1})


if __name__ == '__main__':
    tf.app.run(main=main)
