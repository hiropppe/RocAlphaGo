# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

cimport go


cdef policy_feature_t *allocate_feature():
    cdef policy_feature_t *feature

    feature = <policy_feature_t *>malloc(sizeof(policy_feature_t))
    memset(feature, 0, sizeof(policy_feature_t))

    feature.planes = np.zeros((MAX_POLICY_PLANES, go.pure_board_max), dtype=np.int32)

    feature.search_game = go.allocate_game()
    go.initialize_board(feature.search_game, False)

    return feature


cdef void initialize_feature(policy_feature_t *feature):
    feature.planes[...] = 0
    # Ones: A constant plane filled with 1
    feature.planes[3, :] = 1

    go.initialize_board(feature.search_game, False)


cdef void free_feature(policy_feature_t *feature):
    if feature:
        free(feature)


cdef void update(policy_feature_t *feature, go.game_state_t *game):
    cdef int[:, ::1] F = feature.planes
    cdef go.game_state_t *search_game = feature.search_game

    cdef char current_color = game.current_color
    cdef char other_color = go.FLIP_COLOR(game.current_color)
    cdef int neighbor4[4]
    cdef int pos, npos
    cdef char color, ncolor
    cdef int string_id, nstring_id
    cdef go.string_t *string
    cdef go.string_t *nstring
    cdef int capture_size, self_atari_size, libs_after_move
    cdef int i

    for i in range(go.pure_board_max):
        capture_size = self_atari_size = 0
        pos = go.onboard_pos[i]
        # Stone colour(3): Player stone(0) / opponent stone(1) / empty(2)
        if color == current_color:
            F[0, pos] = 1
        elif color == other_color:
            F[1, pos] = 1
        else:
            F[2, pos] = 1

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
                F[20 + go.MIN(capture_size, 8) - 1, pos] = 1
            # Self-atari size(8): How many of own stones would be captured
            if self_atari_size:
                F[28 + go.MIN(self_atari_size, 8) - 1, pos] = 1
            # Liberties after move(8): Number of liberties after this move is played
            F[36 + go.MIN(libs_after_move, 8) - 1, pos] = 1
            F[46, pos] = 1
        else:
            color = game.board[pos]
            if color:
                string_id = game.string_id[pos]
                string = &game.string[string_id]
                # Turns since(8): How many turns since a move was played
                if game.birth_move[pos]:
                    F[4 + go.MIN(game.moves - game.birth_move[pos], 7), pos] = 1

                # Liberties(8): Number of liberties (empty adjacent points)
                F[12 + go.MIN(string.libs, 8) - 1, pos] = 1
