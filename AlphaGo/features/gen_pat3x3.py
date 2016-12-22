#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import h5py as h5
import numpy as np

import AlphaGo.go as go

import util

try:
    import cPickle as pkl
except:
    import pickle as pkl

from tqdm import tqdm


pat_size = 3
board_size = 5
liberty_cap = 3

WHITE = go.WHITE
BLACK = go.BLACK
EMPTY = go.EMPTY
OUT_OF_INDEX = 3

pos_state = {
    (EMPTY, -1): 0,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (WHITE, 3): 3,
    (BLACK, 1): 4,
    (BLACK, 2): 5,
    (BLACK, 3): 6,
    (OUT_OF_INDEX, -1): 7
}


def get_pos_state_key(gs, pos, board_size):
    """ Return key for state at each intersection of board
        based on stone solor and liberty count
    """
    if 0 <= min(pos) and max(pos) <= board_size-1:
        color = gs.board[pos[0], pos[1]]
        liberty = min(gs.liberty_counts[pos[0], pos[1]], liberty_cap)
    else:
        color = OUT_OF_INDEX
        liberty = -1
    return (color, liberty)


def gen_pattern(input_file, output_file, symmetry=False):
    """ Write possible 3x3 patterns in 5x5 samples
    """
    pat5x5 = h5.File(input_file)
    states = pat5x5['states']
    centers = pat5x5['centers']
    pat3x3 = dict()
    pat_idx = 0

    assert states.len() == centers.len(), 'Invalid input'

    for i in tqdm(range(states.len())):
        gs = go.GameState(size=board_size)
        # Play board state
        for bidx, stone in enumerate(states[i].flatten()):
            action = util.get_position(bidx, board_size)
            if stone == 0:
                gs.do_move(go.PASS_MOVE)
            elif stone == gs.current_player:
                gs.do_move(action)
            else:
                gs.do_move(go.PASS_MOVE)
                gs.do_move(action)

        key = [get_pos_state_key(gs, pos, board_size) for pos in util.get_3x3around(centers[i])]
        pat = np.array([pos_state[index] for index in key]).reshape((pat_size, pat_size))
        if symmetry:
            pat, val, transform = util.get_min_pattern(pat, len(pos_state))
        else:
            val = util.get_pattern_value(pat, len(pos_state))

        if val not in pat3x3:
            pat3x3[val] = pat_idx
            pat_idx += 1

    print('Number of pattern {:d} in {:d} states'.format(len(pat3x3), states.len()))

    with open(output_file, 'w') as f:
        pkl.dump(pat3x3, f)


if __name__ == '__main__':
    try:
        symmetry = int(sys.argv[3])
    except:
        symmetry = 0

    gen_pattern(sys.argv[1], sys.argv[2], symmetry)
