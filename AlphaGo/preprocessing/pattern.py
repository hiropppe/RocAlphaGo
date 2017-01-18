# -*- coding:utf-8 -*-

import numpy as np

import AlphaGo.go as go

from functools32 import lru_cache


liberty_cap = 3

WHITE = go.WHITE
BLACK = go.BLACK
EMPTY = go.EMPTY
OUT_OF_INDEX = 2

vertex_state_map = {
    None: -1,  # OUT_OF_SHAPE
    (EMPTY, -1): 0,
    (WHITE, 1): 1,
    (WHITE, 2): 2,
    (WHITE, 3): 3,
    (BLACK, 1): 4,
    (BLACK, 2): 5,
    (BLACK, 3): 6,
    (OUT_OF_INDEX, -1): 7
}

reverse_vertex_state_map = {
    None: -1,  # OUT_OF_SHAPE
    (EMPTY, -1): 0,
    (BLACK, 1): 1,
    (BLACK, 2): 2,
    (BLACK, 3): 3,
    (WHITE, 1): 4,
    (WHITE, 2): 5,
    (WHITE, 3): 6,
    (OUT_OF_INDEX, -1): 7
}

vertex_state_len = len(vertex_state_map) - 1

size_3x3 = 3
center_3x3 = (int(size_3x3/2), int(size_3x3/2))
base_3x3 = np.array([vertex_state_len] * size_3x3**2).reshape((size_3x3, size_3x3))
exp_3x3 = np.array([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]])

manhattan_d = 2
size_dia = 2 * manhattan_d + 1
center_dia = (int(size_dia/2), int(size_dia/2))
base_dia = np.array([vertex_state_len] * size_dia**2).reshape((size_dia, size_dia))
exp_dia = np.array([[0,  0,  0,  0,  0],
                    [0,  1,  2,  3,  0],
                    [4,  5,  6,  7,  8],
                    [0,  9, 10, 11,  0],
                    [0,  0, 12,  0,  0]])

# encode only color
base_3x3_color = np.array([4] * size_3x3**2).reshape((size_3x3, size_3x3))
base_dia_color = np.array([4] * size_dia**2).reshape((size_dia, size_dia))

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


def get_position(index, board_size):
    """ Conver 1d index to 2d index
    """
    return (index/board_size, index % board_size)


def is_in_manhattan(center, i, j, d=manhattan_d):
    return abs(center[0] - i) + abs(center[1] - j) <= d


def is_around(center, i, j, d=1):
    return max(abs(center[0] - i), abs(center[1] - j)) <= d


def get_index(center, distance_func, distance, board_size, reshape=True):
    pat = np.array([distance_func(center, i, j, distance) and
                    abs(center[0] - i) + abs(center[1] - j) != 0
                    for i in range(board_size)
                    for j in range(board_size)])
    if reshape:
        pat = pat.reshape((board_size, board_size))
    return pat


@lru_cache(maxsize=512)
def get_around_index(center, board_size, distance=1, reshape=True):
    return get_index(center, is_around, distance, board_size, reshape)


@lru_cache(maxsize=512)
def get_diamond_index(center, board_size, distance=2, reshape=True):
    return get_index(center, is_in_manhattan, distance, board_size, reshape)


@lru_cache(maxsize=512)
def get_around(c, d=1):
    """ Return 3x3 indexes around specified center
    """
    return [(c[0] + i, c[1] + j)
            for i in range(-d, d + 1, 1)
            for j in range(-d, d + 1, 1)]


@lru_cache(maxsize=512)
def get_diamond_enclosing_square(c, d=manhattan_d):
    """ Return indexes of square encloding manhattan diamond in d
        from specified center
    """
    return [(c[0] + i, c[1] + j) if is_in_manhattan(c, c[0] + i, c[1] + j, d) else None
            for i in range(-d, d + 1, 1)
            for j in range(-d, d + 1, 1)]


def get_vertex_state_key(gs, pos, board_size, reverse=False):
    """ Return key for state at each intersection of board
        based on stone solor and liberty count
    """
    if pos is None:
        return None

    if 0 <= min(pos) and max(pos) <= board_size - 1:
        color = gs.board[pos[0], pos[1]]
        liberty = min(gs.liberty_counts[pos[0], pos[1]], liberty_cap)
    else:
        color = OUT_OF_INDEX
        liberty = -1
    return (color, liberty)


@lru_cache(maxsize=8192)
def get_vertex_color(gs, pos, board_size, reverse=False):
    """
    """
    if pos is None:
        return -1

    if 0 <= min(pos) and max(pos) <= board_size - 1:
        color = gs.board[pos[0], pos[1]]
        color = -color if reverse else color
    else:
        color = OUT_OF_INDEX
    # adjust range 0-3
    return color + 1


def get_3x3_value(gs, c, symmetric=False, reverse=False):
    """
    """
    if reverse:
        state_map = reverse_vertex_state_map
    else:
        state_map = vertex_state_map

    vtxes = get_around(c)
    pat = np.array([state_map[get_vertex_state_key(
        gs, vtx, gs.size, reverse=reverse)] for vtx in vtxes])
    pat = pat.reshape((size_3x3, size_3x3))
    # clear center
    pat[center_3x3[0], center_3x3[1]] = 0
    if symmetric:
        return min([np.sum(base_3x3 ** exp_3x3 * t(pat)) for t in transformations])
    else:
        return np.sum(base_3x3 ** exp_3x3 * pat)


def get_diamond_value(gs, c, symmetric=True, reverse=False):
    """
    """
    if reverse:
        state_map = reverse_vertex_state_map
    else:
        state_map = vertex_state_map

    vtxes = get_diamond_enclosing_square(c, manhattan_d)
    pat = np.array([state_map[get_vertex_state_key(
        gs, vtx, gs.size, reverse=reverse)] for vtx in vtxes])
    pat = pat.reshape((size_dia, size_dia))
    # clear center
    pat[int(size_dia / 2), int(size_dia / 2)] = 0
    if symmetric:
        return min([np.sum(base_dia ** exp_dia * t(pat)) for t in transformations])
    else:
        return np.sum(base_dia ** exp_dia * pat)


@lru_cache(maxsize=8192)
def get_3x3_color_value(gs, c, symmetric=False, reverse=False):
    """
    """
    vtxes = get_around(c)
    pat = np.array([get_vertex_color(gs, vtx, gs.size, reverse=reverse) for vtx in vtxes])
    pat = pat.reshape((size_3x3, size_3x3))
    # clear center
    pat[center_3x3[0], center_3x3[1]] = 0
    if symmetric:
        return min([np.sum(base_3x3_color ** exp_3x3 * t(pat)) for t in transformations])
    else:
        return np.sum(base_3x3_color ** exp_3x3 * pat)


@lru_cache(maxsize=8192)
def get_diamond_color_value(gs, c, symmetric=True, reverse=False):
    """
    """
    vtxes = get_diamond_enclosing_square(c, manhattan_d)
    pat = np.array([get_vertex_color(gs, vtx, gs.size, reverse=reverse) for vtx in vtxes])
    pat = pat.reshape((size_dia, size_dia))
    # clear center
    pat[center_dia[0], center_dia[1]] = 0
    if symmetric:
        return min([np.sum(base_dia_color ** exp_dia * t(pat)) for t in transformations])
    else:
        return np.sum(base_dia_color ** exp_dia * pat)


def get_3x3_pattern(gs, c, symmetric=False, reverse=False):
    """
    """
    if reverse:
        state_map = reverse_vertex_state_map
    else:
        state_map = vertex_state_map

    vtxes = get_around(c)
    pat = np.array([state_map[get_vertex_state_key(
        gs, vtx, gs.size, reverse=reverse)] for vtx in vtxes])
    pat = pat.reshape((size_3x3, size_3x3))
    if symmetric:
        symmetrics = [t(pat) for t in transformations]
        minidx = np.argmin([np.sum(base_3x3 ** exp_3x3 * sym) for sym in symmetrics])
        pat = symmetrics[minidx]

    idx = get_around_index(center_3x3, size_3x3, 1)
    pat8 = pat[idx].flatten()
    pat64 = np.zeros(pat8.size*8)
    for i, v in enumerate(pat8):
        pat64[int(i*8+v)] = 1
    return pat64


def get_diamond_pattern(gs, c, symmetric=True, reverse=False):
    """
    """
    if reverse:
        state_map = reverse_vertex_state_map
    else:
        state_map = vertex_state_map

    vtxes = get_diamond_enclosing_square(c, manhattan_d)
    pat = np.array([state_map[get_vertex_state_key(
        gs, vtx, gs.size, reverse=reverse)] for vtx in vtxes])
    pat = pat.reshape((size_dia, size_dia))
    # clear center
    pat[int(size_dia / 2), int(size_dia / 2)] = 0
    if symmetric:
        symmetrics = [t(pat) for t in transformations]
        minidx = np.argmin([np.sum(base_dia_color ** exp_dia * sym) for sym in symmetrics])
        pat = symmetrics[minidx]

    idx = get_diamond_index(center_dia, size_dia, manhattan_d)
    pat12 = pat[idx].flatten()
    pat96 = np.zeros(pat12.size*8)
    for i, v in enumerate(pat12):
        pat96[int(i*8+v)] = 1
    return pat96


def get_3x3_color_pattern(gs, c, symmetric=False, reverse=False):
    """
    """
    vtxes = get_around(c)
    pat = np.array([get_vertex_color(gs, vtx, gs.size, reverse=reverse) for vtx in vtxes])
    pat = pat.reshape((size_3x3, size_3x3))
    if symmetric:
        symmetrics = [t(pat) for t in transformations]
        minidx = np.argmin([np.sum(base_3x3_color ** exp_3x3 * sym) for sym in symmetrics])
        pat = symmetrics[minidx]

    idx = get_around_index(center_3x3, size_3x3, 1)
    pat8 = pat[idx].flatten()
    pat32 = np.zeros(pat8.size*4)
    for i, v in enumerate(pat8):
        pat32[int(i*4+v)] = 1
    return pat32


def get_diamond_color_pattern(gs, c, symmetric=True, reverse=False):
    """
    """
    vtxes = get_diamond_enclosing_square(c, manhattan_d)
    pat = np.array([get_vertex_color(gs, vtx, gs.size, reverse=reverse) for vtx in vtxes])
    pat = pat.reshape((size_dia, size_dia))
    # clear center
    pat[center_dia[0], center_dia[1]] = 0
    if symmetric:
        symmetrics = [t(pat) for t in transformations]
        minidx = np.argmin([np.sum(base_dia_color ** exp_dia * sym) for sym in symmetrics])
        pat = symmetrics[minidx]

    idx = get_diamond_index(center_dia, size_dia, manhattan_d)
    pat12 = pat[idx].flatten()
    pat48 = np.zeros(pat12.size*4)
    for i, v in enumerate(pat12):
        pat48[int(i*4+v)] = 1
    return pat48


def get_pattern_value(pat, base, exp):
    return np.sum(base ** exp * pat)


def get_min_pattern(pat, base, exp):
    """ Return min pattern considering symmetry of state board
    """
    transformed = [f(pat) for f in transformations]

    values = [np.sum(base ** exp * t) for t in transformed]
    min_index = np.argmin(values)

    return transformed[min_index], values[min_index], transformations[min_index]


def playout(moves, indexes, center):
    gs = go.GameState(size=indexes.shape[0])
    pidx = 0
    for bidx, do in enumerate(indexes.flatten()):
        if not do:
            continue

        action = get_position(bidx, gs.size)
        stone = moves[pidx]
        pidx += 1
        try:
            if stone in (go.BLACK, go.WHITE):
                gs.do_move(action, stone)
        except go.IllegalMove:
            return None

    try:
        gs.do_move(center, go.BLACK)
        gs.board[center[0], center[1]] = 0
    except go.IllegalMove:
        return None

    return gs.board
