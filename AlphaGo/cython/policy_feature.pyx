# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

cimport go
cimport printer


cdef policy_feature_t *allocate_feature():
    cdef policy_feature_t *feature
    cdef int i

    feature = <policy_feature_t *>malloc(sizeof(policy_feature_t))
    memset(feature, 0, sizeof(policy_feature_t))

    feature.planes = np.zeros((MAX_POLICY_PLANES, go.pure_board_max), dtype=np.int32)

    return feature


cdef void initialize_feature(policy_feature_t *feature):
    feature.planes[...] = 0
    # Ones: A constant plane filled with 1
    feature.planes[3, :] = 1

    for i in range(80):
        go.initialize_board(&feature.search_games[i], False)


cdef void free_feature(policy_feature_t *feature):
    if feature:
        free(feature)


cdef void update(policy_feature_t *feature, go.game_state_t *game):
    cdef int[:, ::1] F = feature.planes

    cdef char current_color = game.current_color
    cdef char other_color = go.FLIP_COLOR(game.current_color)
    cdef int neighbor4[4]
    cdef int pos, npos
    cdef char color, ncolor
    cdef int string_id, nstring_id
    cdef go.string_t *string
    cdef go.string_t *nstring
    cdef int capture_size, self_atari_size, libs_after_move
    cdef short ladder_checked[288] # MAX_STRING
    cdef int i, j
    cdef int ladder_capture, ladder_escape, ladder_x, ladder_y

    F[...] = 0
    go.fill_n_short(ladder_checked, 288, 0)

    for i in range(go.pure_board_max):
        capture_size = self_atari_size = libs_after_move = 0
        pos = go.onboard_pos[i]
        # Stone colour(3): Player stone(0) / opponent stone(1) / empty(2)
        if color == current_color:
            F[0, i] = 1
        elif color == other_color:
            F[1, i] = 1
        else:
            F[2, i] = 1

        if go.is_legal_not_eye(game, pos, current_color):
            go.get_neighbor4(neighbor4, pos)
            for n in range(4):
                npos = neighbor4[n]
                nstring_id = game.string_id[npos]
                if nstring_id:
                    nstring = &game.string[nstring_id]
                    if nstring.libs == 1 and nstring.lib[pos] != 0:
                        if nstring.color == current_color:
                            self_atari_size += nstring.size
                        else:
                            capture_size += nstring.size

                    if nstring.color == current_color:
                        libs_after_move += (nstring.libs - 1)
                    elif nstring.libs == 1 and nstring.lib[pos] == 0:
                        libs_after_move += 1
                else:
                    libs_after_move += 1
            # Capture size(8): How many opponent stones would be captured
            if capture_size:
                F[20 + go.MIN(capture_size, 8) - 1, i] = 1
            # Self-atari size(8): How many of own stones would be captured
            if self_atari_size:
                F[28 + go.MIN(self_atari_size, 8) - 1, i] = 1
            # Liberties after move(8): Number of liberties after this move is played
            F[36 + go.MIN(libs_after_move, 8) - 1, i] = 1
            F[46, i] = 1
        else:
            color = game.board[pos]
            if color:
                string_id = game.string_id[pos]
                string = &game.string[string_id]
                # Turns since(8): How many turns since a move was played
                if game.birth_move[pos]:
                    F[4 + go.MIN(game.moves - game.birth_move[pos], 7), i] = 1

                # Liberties(8): Number of liberties (empty adjacent points)
                F[12 + go.MIN(string.libs, 8) - 1, i] = 1

                if not ladder_checked[string_id]:
                    # Ladder capture(1): Whether a move at this point is a successful ladder capture
                    if string.libs == 2 and string.color == other_color:
                        for j in range(2):
                            if j == 0:
                                ladder_capture = string.lib[0]
                                ladder_escape = string.lib[string.lib[0]]
                            else:
                                ladder_capture = string.lib[string.lib[0]]
                                ladder_escape = string.lib[0]

                            if is_ladder_capture(game, string_id,
                                                 ladder_capture, ladder_escape,
                                                 feature.search_games, 0):
                                ladder_x = go.CORRECT_X(ladder_capture, go.board_size, go.OB_SIZE)
                                ladder_y = go.CORRECT_Y(ladder_capture, go.board_size, go.OB_SIZE)
                                F[44, go.POS(ladder_x, ladder_y, go.pure_board_size)] = 1
                    # Ladder escape(1): Whether a move at this point is a successful ladder escape
                    elif string.libs == 1 and string.color == current_color:
                        if is_ladder_escape(game, string_id, string.lib[0], feature.search_games, 0):
                            ladder_x = go.CORRECT_X(string.lib[0], go.board_size, go.OB_SIZE)
                            ladder_y = go.CORRECT_Y(string.lib[0], go.board_size, go.OB_SIZE)
                            F[45, go.POS(ladder_x, ladder_y, go.pure_board_size)] = 1
                    ladder_checked[string_id] = True


cdef bint is_ladder_capture(go.game_state_t *game, int string_id, int pos, int other_pos,
                            go.game_state_t search_games[80], int depth):
    cdef go.game_state_t *ladder_game = &search_games[depth]
    cdef go.string_t *string
    cdef char capture_color = game.current_color

    if not go.is_legal(game, pos, capture_color):
        return False

    if depth >= 80:
        return False

    go.copy_game(ladder_game, game)
    go.do_move(ladder_game, pos)

    string = &ladder_game.string[string_id]
    if is_ladder_escape(ladder_game, string_id, other_pos, search_games, depth+1):
        return False
    else:
        return True


cdef bint is_ladder_escape(go.game_state_t *game, int string_id, int pos,
                           go.game_state_t search_games[80], int depth):
    cdef go.game_state_t *ladder_game = &search_games[depth]
    cdef go.string_t *string
    cdef char escape_color = game.current_color
    cdef int neighbor_id
    cdef go.string_t *neighbor_string
    cdef bint capture_neighbor = False
    cdef int j
    cdef int ladder_capture, ladder_escape

    if not go.is_legal(game, pos, escape_color):
        return False

    if depth >= 80:
        return False

    # capture atari neighbor if exists
    string = &game.string[string_id]
    neighbor_id = string.neighbor[0]
    while neighbor_id != go.neighbor_end and not capture_neighbor:
        neighbor = &game.string[neighbor_id]
        if neighbor.libs == 1 and go.is_legal(game, neighbor.lib[0], escape_color):
            go.copy_game(ladder_game, game)
            go.do_move(ladder_game, neighbor.lib[0])
            capture_neighbor = True
        else:
            neighbor_id = string.neighbor[neighbor_id]

    if not capture_neighbor:
        go.copy_game(ladder_game, game)
        go.do_move(ladder_game, pos)

    string = &ladder_game.string[string_id]
    if string.libs == 1:
        return False
    elif string.libs >= 3:
        return True
    else:
        for j in range(2):
            if j == 0:
                ladder_capture = string.lib[0]
                ladder_escape = string.lib[string.lib[0]]
            else:
                ladder_capture = string.lib[string.lib[0]]
                ladder_escape = string.lib[0]

            if is_ladder_capture(ladder_game, string_id,
                                 ladder_capture, ladder_escape,
                                 search_games, depth+1):
                return False
        return True
