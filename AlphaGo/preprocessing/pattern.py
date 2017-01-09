# -*- coding:utf-8 -*-

import numpy as np

import AlphaGo.go as go


liberty_cap = 3

WHITE = go.WHITE
BLACK = go.BLACK
EMPTY = go.EMPTY
OUT_OF_INDEX = 3

vertex_state_map = {
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

vertex_state_len = len(vertex_state_map)-1

size_3x3 = 3
base_3x3 = np.array([vertex_state_len]*size_3x3**2).reshape((size_3x3, size_3x3))
exp_3x3 = np.array([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]])

manhattan_d = 2
size_dia = 2*manhattan_d+1
base_dia = np.array([vertex_state_len]*size_dia**2).reshape((size_dia, size_dia))
exp_dia = np.array([[0,  0,  0,  0,  0],
                    [0,  1,  2,  3,  0],
                    [4,  5,  6,  7,  8],
                    [0,  9, 10, 11,  0],
                    [0,  0, 12,  0,  0]])

transformations = [
    lambda board: board,
    lambda board: np.rot90(board, 1),
    lambda board: np.rot90(board, 2),
    lambda board: np.rot90(board, 3),
    lambda board: np.fliplr(board),
    lambda board: np.flipud(board),
    lambda board: np.transpose(board),
    lambda board: np.fliplr(np.rot90(board, 1))
]


def is_in_manhattan(center, i, j, d=manhattan_d):
    return abs(center[0]-i) + abs(center[1]-j) <= d


def is_around(center, i, j, d=1):
    return max(abs(center[0]-i), abs(center[1]-j)) <= d


def get_index(center, distance_func, distance, board_size):
    return np.array([distance_func(center, i, j, distance) and
                    abs(center[0]-i) + abs(center[1]-j) != 0
                    for i in range(board_size)
                    for j in range(board_size)]).reshape((board_size, board_size))


def get_around_index(center, board_size, distance=1):
    return get_index(center, is_around, distance, board_size)


def get_diamond_index(center, board_size, distance=2):
    return get_index(center, is_in_manhattan, distance, board_size)


def get_around(c, d=1):
    """ Return 3x3 indexes around specified center
    """
    return tuple([(c[0]+i, c[1]+j)
                  for i in range(-d, d+1, 1)
                  for j in range(-d, d+1, 1)])


def get_diamond_enclosing_square(c, d=manhattan_d):
    """ Return indexes of square encloding manhattan diamond in d
        from specified center
    """
    return tuple([(c[0]+i, c[1]+j) if is_in_manhattan(c, c[0]+i, c[1]+j, d) else None
                  for i in range(-d, d+1, 1)
                  for j in range(-d, d+1, 1)])


def get_vertex_state_key(gs, pos, board_size, reverse=False):
    """ Return key for state at each intersection of board
        based on stone solor and liberty count
    """
    if pos is None:
        return None

    if 0 <= min(pos) and max(pos) <= board_size-1:
        color = gs.board[pos[0], pos[1]]
        # Pattern's center is always BLACK otherwise diamond center has opposite color
        if reverse:
            if gs.current_player == go.BLACK:
                color = -color
        else:
            if gs.current_player == go.WHITE:
                color = -color
        liberty = min(gs.liberty_counts[pos[0], pos[1]], liberty_cap)
    else:
        color = OUT_OF_INDEX
        liberty = -1
    return (color, liberty)


def get_3x3_value(gs, c):
    """
    """
    vtxes = get_around(c)
    pat = np.array([vertex_state_map[get_vertex_state_key(gs, vtx, gs.size)] for vtx in vtxes])
    pat = pat.reshape((size_3x3, size_3x3))
    return np.sum(base_3x3 ** exp_3x3 * pat)


def get_diamond_value(gs):
    """
    """
    vtxes = get_diamond_enclosing_square(gs.history[-1], manhattan_d)
    pat = np.array([vertex_state_map[get_vertex_state_key(gs, vtx, gs.size, reverse=True)] for vtx in vtxes])
    pat = pat.reshape((size_dia, size_dia))
    # clear center
    pat[int(size_dia/2), int(size_dia/2)] = 0
    return min([np.sum(base_dia ** exp_dia * t(pat)) for t in transformations])
