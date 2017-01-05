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


board_size = 5
liberty_cap = 3

WHITE = go.WHITE
BLACK = go.BLACK
EMPTY = go.EMPTY
OUT_OF_INDEX = 3

pos_state = {
    None: -1,
    (EMPTY, -1): 0,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (WHITE, 3): 3,
    (BLACK, 1): 4,
    (BLACK, 2): 5,
    (BLACK, 3): 6,
    (OUT_OF_INDEX, -1): 7
}

pos_state_len = len(pos_state)-1


def get_pos_state_key(gs, pos, board_size):
    """ Return key for state at each intersection of board
        based on stone solor and liberty count
    """
    if pos is None:
        return None

    if 0 <= min(pos) and max(pos) <= board_size-1:
        color = gs.board[pos[0], pos[1]]
        liberty = min(gs.liberty_counts[pos[0], pos[1]], liberty_cap)
    else:
        color = OUT_OF_INDEX
        liberty = -1
    return (color, liberty)


def gen_pattern(input_file, output_file, pat_filter, symmetry=False):
    """ Write possible 3x3 patterns in 5x5 samples
    """
    pat5x5 = h5.File(input_file)
    states = pat5x5['states']
    centers = pat5x5['centers']
    pat3x3 = dict()
    pat_idx = 0

    assert states.len() == centers.len(), 'Invalid input'

    base = None
    power = None
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

        state_keys = [get_pos_state_key(gs, pos, board_size) for pos in pat_filter(centers[i])]
        pat_size = int(np.sqrt(len(state_keys)))
        pat = np.array([pos_state[state_key] for state_key in state_keys]) \
                .reshape((pat_size, pat_size))

        if base is None:
            base = np.array([pos_state_len]*pat_size**2).reshape((pat_size, pat_size))

        if power is None:
            p = 0
            power = []
            for s in pat.flatten():
                if s == -1:
                    power.append(0)
                else:
                    power.append(p)
                    p += 1
            power = np.array(power).reshape((pat_size, pat_size))

        if symmetry:
            pat, val, transform = util.get_min_pattern(pat, base, power)
        else:
            val = util.get_pattern_value(pat, base, power)

        if val not in pat3x3:
            pat3x3[val] = pat_idx
            pat_idx += 1

    print('Number of pattern {:d} in {:d} states'.format(len(pat3x3), states.len()))

    with open(output_file, 'w') as f:
        pkl.dump(pat3x3, f)


if __name__ == '__main__':
    try:
        pat = sys.argv[3]
        if pat == 'mht':
            pat_filter = lambda c: util.get_diamond_enclosing_square(c, int(sys.argv[4]))
        else:
            pat_filter = lambda c: util.get_around(c, int(sys.argv[4]))
    except:
        pat_filter = lambda c: util.get_around(c, 1)

    try:
        symmetry = int(sys.argv[5])
    except:
        symmetry = 0

    gen_pattern(sys.argv[1], sys.argv[2], pat_filter, symmetry)
