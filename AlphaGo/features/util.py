# -*- coding:utf-8 -*-

import numpy as np

import AlphaGo.go as go


transforms = [
    lambda board: board,
    lambda board: np.rot90(board, 1),
    lambda board: np.rot90(board, 2),
    lambda board: np.rot90(board, 3),
    lambda board: np.fliplr(board),
    lambda board: np.flipud(board),
    lambda board: np.transpose(board),
    lambda board: np.fliplr(np.rot90(board, 1))
]


def get_position(index, board_size):
    """ Conver 1d index to 2d index
    """
    return (index/board_size, index % board_size)


def is_in_manhattan(center, i, j, d):
    return abs(center[0]-i) + abs(center[1]-j) <= d


def get_around(c, d):
    """ Return 3x3 indexes around specified center
    """
    return tuple([(c[0]+i, c[1]+j)
                  for i in range(-d, d+1, 1)
                  for j in range(-d, d+1, 1)])


def get_diamond_enclosing_square(c, d):
    """ Return indexes of square encloding manhattan diamond in d
        from specified center
    """
    return tuple([(c[0]+i, c[1]+j) if is_in_manhattan(c, c[0]+i, c[1]+j, d) else None
                  for i in range(-d, d+1, 1)
                  for j in range(-d, d+1, 1)])


def get_pattern_value(board, base, power):
    value = np.sum(base ** power * board)
    return value


def get_min_pattern(board, base, power):
    """ Return min pattern considering symmetry of state board
    """
    symmetries = [f(board) for f in transforms]

    values = [np.sum(base ** power * pat) for pat in symmetries]
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

        base_val = 3
        base = np.array([base_val]*gs.size**2).reshape((gs.size, gs.size))
        power = np.arange(gs.size**2).reshape((gs.size, gs.size))
        if symmetry:
            min_board, min_value, min_func = get_min_pattern(gs.board+1, base, power)

            # get center after transform
            tmp = np.zeros(gs.board.shape)
            tmp[center[0], center[1]] = 1
            tmp = min_func(tmp)
            nonzero = tmp.nonzero()
            new_center = zip(nonzero[0], nonzero[1])[0]
            return min_board-1, min_value, new_center
        else:
            return gs.board, get_pattern_value(gs.board+1, base, power), center
    except go.IllegalMove:
        return None
