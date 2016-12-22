# -*- coding:utf-8 -*-

import numpy as np

import AlphaGo.go as go


transforms = [
    lambda board: board,
    lambda board: np.rot90(board),
    lambda board: np.rot90(np.rot90(board)),
    lambda board: np.rot90(np.rot90(np.rot90(board))),
    lambda board: np.fliplr(board),
    lambda board: np.flipud(board),
    lambda board: np.transpose(board),
    lambda board: np.rot90(np.fliplr(board))
]


def get_position(index, board_size):
    """ Conver 1d index to 2d index
    """
    return (index/board_size, index % board_size)


def get_3x3around(c):
    """ Return 3x3 indexes around specified center
    """
    return ((c[0]-1, c[1]-1), (c[0]-1, c[1]), (c[0]-1, c[1]+1),
            (c[0], c[1]-1), (c[0], c[1]), (c[0], c[1]+1),
            (c[0]+1, c[1]), (c[0]+1, c[1]), (c[0], c[1]+1))


def get_pattern_value(board, base):
    size = board.shape[0]
    base_array = np.array([base]*size**2).reshape((size, size))
    power_array = np.arange(size**2).reshape((size, size))
    value = np.sum(base_array ** power_array * (board + 1))
    return value


def get_min_pattern(board, base):
    """ Return min pattern considering symmetry of state board
    """
    symmetries = [f(board) for f in transforms]

    size = board.shape[0]
    base_array = np.array([base]*size**2).reshape((size, size))
    power_array = np.arange(size**2).reshape((size, size))
    values = [np.sum(base_array ** power_array * (pat + 1)) for pat in symmetries]
    min_index = np.argmin(values)

    return symmetries[min_index], values[min_index], transforms[min_index]


def playout(moves, indexes, center, symmetry=False):
    assert len(moves) == sum(indexes), 'Invalid playout input'
    gs = go.GameState(size=int(np.sqrt(len(indexes))))
    pidx = 0
    for bidx, do in enumerate(indexes):
        if not do:
            continue

        action = get_position(bidx, gs.size)
        stone = moves[pidx]
        pidx += 1
        try:
            if stone == 0:
                gs.do_move(go.PASS_MOVE)
            elif stone == gs.current_player:
                gs.do_move(action)
            else:
                gs.do_move(go.PASS_MOVE)
                gs.do_move(action)
        except go.IllegalMove:
            return None

    if gs.current_player == go.WHITE:
        gs.do_move(go.PASS_MOVE)

    try:
        gs.do_move(center)
        # clear center
        gs.board[center[0], center[1]] = 0
        if symmetry:
            min_board, min_value, min_func = get_min_pattern(gs.board, 3)

            # get center after transform
            tmp = np.zeros(gs.board.shape)
            tmp[center[0], center[1]] = 1
            tmp = min_func(tmp)
            nonzero = tmp.nonzero()
            new_center = zip(nonzero[0], nonzero[1])[0]
            return min_board, min_value, new_center
        else:
            return gs.board, get_pattern_value(gs.board, 3), center
    except go.IllegalMove:
        return None
