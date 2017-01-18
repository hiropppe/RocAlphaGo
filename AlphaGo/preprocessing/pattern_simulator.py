#!/usr/bin/env python
# -*- coding:utf-8 -*-

import itertools
import os
import re
import sys

import numpy as np
import h5py as h5

from tqdm import tqdm

import pattern as ptn


class PatternSimulator(object):

    def __init__(self, board_size=5):
        self.board_size = board_size
        self.state_value_set = set()

    def gen_patterns(self, output_file, centers, pattern_func):
        print('Simulate moves around centers')
        print('  Centers: {}'.format(centers))
        print('  Pattern: {}'.format(str(pattern_func)))

        tmp_file = os.path.join(output_file + '.tmp')
        h5f = h5.File(tmp_file, 'w')

        self.state_value_set.clear()

        self.states = h5f.require_dataset(
            'states',
            dtype=np.int8,
            shape=(1, self.board_size, self.board_size),
            maxshape=(None, self.board_size, self.board_size),
            exact=False,
            compression='lzf')
        self.centers = h5f.require_dataset(
            'centers',
            dtype=np.int8,
            shape=(1, 2),
            maxshape=(None, 2),
            exact=False,
            compression='lzf')

        self.nb_play = 0
        self.data_idx = 0

        for center in centers:
            self.gen_pattern_by_center(center, pattern_func)

        print(('Number of pattern: {:d} Save pattern: {:d}'
               .format(self.nb_play, self.states.len())))

        h5f.close()
        os.rename(tmp_file, output_file)

    def add_pattern(self, board, center):
        if board is not None:
            if self.data_idx >= len(self.states):
                self.states.resize((self.data_idx + 1, self.board_size, self.board_size))
                self.centers.resize((self.data_idx + 1, 2))
            self.states[self.data_idx] = board
            self.centers[self.data_idx] = center
            return True

    def gen_pattern_by_center(self, center, pattern_func):
        # Mark put stone or not around center
        play_index = pattern_func(center, self.board_size)
        play_num = np.sum(play_index)

        progress_bar = tqdm(total=3**play_num)
        progress_bar.set_description(str(center))
        for p in itertools.product((-1, 0, 1), repeat=play_num):
            board = ptn.playout(p, play_index, center)
            successful = self.add_pattern(board, center)
            if successful:
                self.data_idx += 1

            self.nb_play += 1
            progress_bar.update(1)
        progress_bar.close()


if __name__ == '__main__':
    board_size = 5
    try:
        pattern_func = sys.argv[2]
        if pattern_func == 'mht':
            pattern_func = ptn.get_diamond_index
        else:
            pattern_func = ptn.get_around_index
    except:
        pattern_func = ptn.get_around_index

    try:
        centers = [tuple([int(e.split(',')[0]), int(e.split(',')[1])])
                   for e in re.findall(r'\d\s*,\s*\d', sys.argv[4])]
    except:
        centers = [(i, j) for i in xrange(board_size) for j in xrange(board_size)]

    pg = PatternSimulator(board_size=board_size)
    pg.gen_patterns(sys.argv[1], centers, pattern_func)
