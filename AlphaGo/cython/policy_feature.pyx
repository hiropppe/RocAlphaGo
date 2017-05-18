# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

cimport go


cdef void update(int[:, ::1] F, go.game_state_t *game):
    cdef int other = go.FLIP_COLOR(game.current_color)
    cdef int i, pos
    cdef int string_id
    cdef char color
    cdef go.string_t *string

    for i in range(go.pure_board_max):
        pos = go.onboard_pos[i]
        color = game.board[pos]
        # Stone colour: Player stone(0) / opponent stone(1) / empty(2)
        if color == game.current_color:
            F[0, pos] = 1
        elif color == other:
            F[1, pos] = 1
        else:
            F[2, pos] = 1

        # Turns since: How many turns since a move was played
        if game.birth_move[pos]:
            F[4 + go.MIN(game.moves - game.birth_move[pos], 7)] = 1

        # Liberties: Number of liberties (empty adjacent points)
        if color != go.S_EMPTY:
            string_id = game.string_id[pos]
            string = &game.string[string_id]
            F[12 + go.MIN(string.libs, 7) - 1] = 1
