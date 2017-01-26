# -*- coding: utf-8 -*-
#cython: boundscheck=False
"""
#cython: wraparound=False
"""

import numpy as np

cimport numpy as np

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair

DTYPE = np.int

ctypedef np.int_t DTYPE_t


cdef class RolloutFeature(object):
    cdef int b_size
    cdef public int n_position

    cdef int n_response
    cdef int n_save_atari
    cdef int n_neighbor
    cdef int n_feature

    cdef int n_3x3
    cdef int n_12d

    cdef int ix_response
    cdef int ix_save_atari
    cdef int ix_neighbor
    cdef int ix_3x3
    cdef int ix_12d

    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_CACHE

    cdef pair[vector[int], vector[int]] _prev_neighbors

    def __init__(self, b_size, pat3x3_file=None, pat12d_file=None):
        self.b_size = b_size
        self.n_position = self.b_size * self.b_size

        self.n_response = 1
        self.n_save_atari = 1
        self.n_neighbor = 8

        if pat3x3_file or pat12d_file:
            try:
                import cPickle as pkl
            except:
                import pickle as pkl

            if pat3x3_file:
                self.pat3x3 = pkl.load(open(pat3x3_file))
                self.n_3x3 = len(self.pat3x3)

            if pat12d_file:
                self.pat12d = pkl.load(open(pat12d_file))
                self.n_12d = len(self.pat12d)
        else:
            self.n_3x3 = 69338
            self.n_12d = 32207

        self.n_feature = self.n_response + self.n_save_atari + self.n_neighbor + self.n_3x3 + self.n_12d

        self.ix_response = 0
        self.ix_save_atari = self.ix_response + self.n_response
        self.ix_neighbor = self.ix_save_atari + self.n_save_atari
        self.ix_3x3 = self.ix_neighbor + self.n_neighbor
        self.ix_12d = self.ix_3x3 + self.n_3x3

        self._create_neighbor8_cache()

        cdef vector[int] ix
        cdef vector[int] iy
        self._prev_neighbors.first = ix
        self._prev_neighbors.second = iy

    def _create_neighbor8_cache(self):
        for x in range(self.b_size):
            for y in range(self.b_size):
                (ix, iy) = self._get_neighbor8((x, y))
                position = x*self.b_size+y
                self.__NEIGHBOR8_CACHE[position].first = ix
                self.__NEIGHBOR8_CACHE[position].second = iy

    def _get_neighbor8(self, center):
        (x, y) = center
        neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                     (x,   y-1),           (x,   y+1),
                     (x+1, y-1), (x+1, y), (x+1, y+1)]
        xy = np.array([[nx*self.b_size + ny, self.ix_neighbor+i]
                       for i, (nx, ny) in enumerate(neighbors) if self._on_board((nx, ny))])
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _on_board(self, position):
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.b_size and y < self.b_size

    def update(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        #cdef int prev_position
        if state.history:
            prev_move = state.history[-1]
            prev_position = prev_move[0]*self.b_size+prev_move[1]
        else:
            prev_position = -1

        self.update_neighbors(prev_position, feature)
        self.update_save_atari(state, feature)

    def update_neighbors(self, prev_position, np.ndarray[DTYPE_t, ndim=2] feature):
        # clear previous neighbors
        if not self._prev_neighbors.first.empty():
            prev_nx = self._prev_neighbors.first
            prev_ny = self._prev_neighbors.second
            feature[prev_nx, prev_ny] = 0

        if 0 < prev_position:
            # set new neighbors
            prev = self.__NEIGHBOR8_CACHE[prev_position]
            nx = prev.first
            ny = prev.second
            feature[nx, ny] = 1
            self._prev_neighbors.first = nx
            self._prev_neighbors.second = ny

    def update_save_atari(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        feature[:, self.ix_save_atari] = 0
        for (x, y) in state.last_liberty_cache[state.current_player]:
            if len(state.liberty_sets[x][y]) > 1:
                feature[x*self.b_size + y, self.ix_save_atari] = 1
            else:
                for (nx, ny) in state._neighbors((x, y)):
                    if (state.board[nx, ny] == state.current_player
                       and state.liberty_counts[nx, ny] > 1):
                        feature[x*self.b_size + y, self.ix_save_atari] = 1


def timeit():
    """ Testing performance
    """
    import random
    import sys
    import time
    import traceback
    import AlphaGo.go as go

    b_size = 19
    rf = RolloutFeature(b_size)
    
    cdef np.ndarray[DTYPE_t, ndim=2] F
    F = np.zeros((b_size**2, rf.n_position), dtype=DTYPE)

    elapsed_s = list()
    num_error = 0
    for i in xrange(100):
        gs = go.GameState()

        empties = range(361)
        for j in range(200):
            move = random.choice(empties)
            empties.remove(move)
            x = int(move / 19)
            y = int(move % 19)
            try:
                gs.do_move((x, y))
                s = time.time()
                rf.update(gs, F)
                elapsed_s.append(time.time() - s)
            except go.IllegalMove:
                # sys.stderr.write("{}\n".format(traceback.format_exc()))
                num_error += 1
                continue
    print("Avg. {:.3f}us Err. {:d}".format(np.mean(elapsed_s)*1000*1000, num_error))
