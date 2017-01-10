#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import h5py as h5

import AlphaGo.go as go
import AlphaGo.preprocessing.pattern as ptn

try:
    import cPickle as pkl
except:
    import pickle as pkl

from tqdm import tqdm


board_size = 5


def gen_pattern(input_file, output_file, f_pat_value, symmetric=False):
    """ Write possible 3x3 patterns in 5x5 samples
    """
    board = h5.File(input_file)
    states = board['states']
    centers = board['centers']
    pat_dict = dict()
    pat_onehot_idx = 0

    assert states.len() == centers.len(), 'Invalid input'

    for i in tqdm(range(states.len())):
        gs = go.GameState(size=states[i].shape[0])
        # Replay board state
        for bidx, stone in enumerate(states[i].flatten()):
            action = ptn.get_position(bidx, board_size)
            if stone == 0:
                gs.do_move(go.PASS_MOVE)
            elif stone == gs.current_player:
                gs.do_move(action)
            else:
                gs.do_move(go.PASS_MOVE)
                gs.do_move(action)

        val = f_pat_value(gs, centers[i], symmetric, reverse=False)

        if val not in pat_dict:
            pat_dict[val] = pat_onehot_idx
            pat_onehot_idx += 1

    print('Number of pattern {:d} in {:d} states'.format(len(pat_dict), states.len()))

    with open(output_file, 'w') as f:
        pkl.dump(pat_dict, f)


if __name__ == '__main__':
    try:
        pat = sys.argv[3]
        if pat == 'dia12':
            f_pat_value = ptn.get_diamond_value
        elif pat == 'dia12c':
            f_pat_value = ptn.get_diamond_color_value
        elif pat == '3x3':
            f_pat_value = ptn.get_3x3_value
        elif pat == '3x3c':
            f_pat_value = ptn.get_3x3_color_value
        else:
            sys.stderr.write('Illegal pattern %s\n'.format(sys.argv[3]))
            sys.exit(1)
    except:
        f_pat_value = ptn.get_3x3_value

    try:
        symmetric = int(sys.argv[4])
    except:
        symmetric = 0

    gen_pattern(sys.argv[1], sys.argv[2], f_pat_value, symmetric)
