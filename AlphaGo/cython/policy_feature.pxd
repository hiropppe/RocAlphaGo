# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

cimport go


cdef void update(int[:, ::1], go.game_state_t *game)
