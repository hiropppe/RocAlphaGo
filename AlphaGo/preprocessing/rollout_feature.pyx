# -*- coding: utf-8 -*-
#cython: boundscheck=False
"""
#cython: wraparound=False
"""
import numpy as np

cimport numpy as np

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map


DTYPE = np.int

ctypedef np.int_t DTYPE_t


cdef enum:
    color_shift = 2
    max_liberty_count = 3


cdef class RolloutFeature(object):
    cdef public int b_size
    cdef public int n_position
    cdef public int n_feature

    cdef public unordered_map[long long, int] pattern_3x3
    cdef public unordered_map[long long, int] pattern_12d

    cdef public int ix_response
    cdef public int ix_save_atari
    cdef public int ix_neighbor
    cdef public int ix_3x3
    cdef public int ix_12d

    cdef int n_response
    cdef int n_save_atari
    cdef int n_neighbor

    cdef int n_3x3
    cdef int n_12d

    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR4_INDEX_CACHE
    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_INDEX_CACHE
    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_FEATURE_CACHE

    cdef pair[vector[int], vector[int]] _prev_neighbors

    cdef np.ndarray prev_board

    def __init__(self, b_size, f_pattern_3x3=None, f_pattern_12d=None):
        self.b_size = b_size
        self.n_position = self.b_size * self.b_size

        self.n_response = 1
        self.n_save_atari = 1
        self.n_neighbor = 8
        self.n_3x3 = 4500
        self.n_12d = 2300

        if f_pattern_3x3 or f_pattern_12d:
            import ast
            if f_pattern_3x3:
                with open(f_pattern_3x3) as f:
                    self.pattern_3x3 = ast.literal_eval(f.read())
            if f_pattern_12d:
                with open(f_pattern_12d) as f:
                    self.pattern_12d = ast.literal_eval(f.read())

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

        self.prev_board = np.zeros((b_size, b_size), dtype=np.int)

    def _create_neighbor8_cache(self):
        for x in xrange(self.b_size):
            for y in xrange(self.b_size):
                position = x*self.b_size+y
                (fx, fy) = self._get_neighbor8_feature((x, y))
                self.__NEIGHBOR8_FEATURE_CACHE[position].first = fx
                self.__NEIGHBOR8_FEATURE_CACHE[position].second = fy
                # cache neighbor4
                (ix4, iy4) = self._get_neighbor4_index((x, y))
                self.__NEIGHBOR4_INDEX_CACHE[position].first = ix4
                self.__NEIGHBOR4_INDEX_CACHE[position].second = iy4
                # cache neighbor8
                (ix8, iy8) = self._get_neighbor8_index((x, y))
                self.__NEIGHBOR8_INDEX_CACHE[position].first = ix8
                self.__NEIGHBOR8_INDEX_CACHE[position].second = iy8

    def _get_neighbor4_index(self, center):
        """Returns neighbor8 index pair ([r0, r1, r2, r3], [c0, c1, c2, c3])
        """
        neighbors = self._get_neighbor4(center)
        xy = np.array(neighbors)
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _get_neighbor8_index(self, center):
        """Returns neighbor8 index pair ([r0, r1, .., r7], [c0, c1, .., c7])
        """
        neighbors = self._get_neighbor8(center)
        xy = np.array(neighbors)
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _get_neighbor8_feature(self, center):
        """Returns nenighbor feature index ([np0, np1, .., np7], [ni0, ni1, .., ni7])
        """
        neighbors = self._get_neighbor8(center)
        xy = np.array([[nx*self.b_size + ny, self.ix_neighbor+i]
                       for i, (nx, ny) in enumerate(neighbors) if self._on_board((nx, ny))])
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _get_neighbor4(self, center):
        (x, y) = center
        return [          (x-1, y),
                (x, y-1),           (x, y+1),
                          (x+1, y)          ]

    def _get_neighbor8(self, center):
        (x, y) = center
        return [(x-1, y-1), (x-1, y), (x-1, y+1),
                (x,   y-1),           (x,   y+1),
                (x+1, y-1), (x+1, y), (x+1, y+1)]

    def _on_board(self, position):
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.b_size and y < self.b_size

    def update(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        """
        """
        cdef int px, py, prev_position

        if state.history:
            prev_move = state.history[-1]
            if prev_move:
                (px, py) = prev_move
                prev_position = px*self.b_size+py
        else:
            prev_position = -1

        self.update_neighbors(prev_position, feature)
        self.update_save_atari(state, feature)
        self.update_non_response_3x3(state, feature)

    def update_neighbors(self, prev_position, np.ndarray[DTYPE_t, ndim=2] feature):
        """
        """
        # clear previous neighbors
        if not self._prev_neighbors.first.empty():
            prev_nx = self._prev_neighbors.first
            prev_ny = self._prev_neighbors.second
            feature[prev_nx, prev_ny] = 0

        if 0 <= prev_position:
            # set new neighbors
            prev = self.__NEIGHBOR8_FEATURE_CACHE[prev_position]
            nx = prev.first
            ny = prev.second
            feature[nx, ny] = 1
            self._prev_neighbors.first = nx
            self._prev_neighbors.second = ny

    def update_save_atari(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        """
        """
        feature[:, self.ix_save_atari] = 0
        for (x, y) in state.last_liberty_cache[state.current_player]:
            if len(state.liberty_sets[x][y]) > 1:
                feature[x*self.b_size + y, self.ix_save_atari] = 1
            else:
                for (nx, ny) in state._neighbors((x, y)):
                    if (state.board[nx, ny] == state.current_player
                       and state.liberty_counts[nx, ny] > 1):
                        feature[x*self.b_size + y, self.ix_save_atari] = 1

    def update_non_response_3x3(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        """
        """
        cdef vector[int] updated_positions

        # Get updated positions by previous move
        updated_positions = self.get_updated_positions(state)

        # Update pattern around positions
        self.update_3x3_around(state, updated_positions, feature)

    def get_updated_positions(self, state):
        """
        """
        cdef vector[int] updated_positions
        cdef int difference_size, gx, gy

        # For each first move check only last move
        # otherwise we obtain vertexes which affects neighbor 3x3 pattern
        # from board difference to previous board
        if not state.history or not state.history[-1]:
            return updated_positions

        if len(state.history) <= 2:
            board_diff = [state.history[-1]]
        else:
            board_diff = np.where(state.board-self.prev_board != 0) 
            board_diff = zip(board_diff[0], board_diff[1])
        # Save board
        self.prev_board = state.board.copy()

        # Obtain group (ren) of updated vertex
        # which color or liberty count may change
        difference_size = len(board_diff)
        for i in xrange(difference_size):
            (dx, dy) = board_diff[i]
            group = state.group_sets[dx][dy]
            if group:
                for (gx, gy) in group:
                    updated_positions.push_back(gx*self.b_size+gy)

        return updated_positions

    def update_3x3_around(self, state, vector[int] updated_positions, np.ndarray[DTYPE_t, ndim=2] feature):
        """
        """
        cdef long long pattern_key
        cdef int pattern_id
        cdef int cp, nx, ny, fx, fy
        
        # Update pattern around updated vertexes
        for cp in updated_positions:
            neighbor8_index = self.__NEIGHBOR8_INDEX_CACHE[cp]
            ix8 = neighbor8_index.first
            iy8 = neighbor8_index.second
            centers = zip(ix8, iy8)
            for j in xrange(8):
                (nx, ny) = centers[j]
                pattern_key = self.get_3x3_key(state, (nx, ny))
                if pattern_key >= 0:
                    if(self.pattern_3x3.end() != self.pattern_3x3.find(pattern_key)):
                        pattern_id = self.pattern_3x3[pattern_key]
                        fx = nx*self.b_size+ny
                        fy = self.ix_3x3 + pattern_id
                        feature[fx, fy] = 1

    def get_3x3_key(self, state, center):
        """
        """
        cdef long long pattern = 0
        cdef int x, y, p
        cdef vector[int] ix_3x3, iy_3x3
        cdef vector[int] color_3x3, liberty_3x3

        (x, y) = center
        # position is to close to the edge
        if x < 1 or y < 1 or x >= self.b_size - 1 or y >= self.b_size - 1:
            return -1

        p = x * self.b_size + y
        ix_3x3 = self.__NEIGHBOR8_INDEX_CACHE[p].first
        iy_3x3 = self.__NEIGHBOR8_INDEX_CACHE[p].second

        pattern = state.current_player + color_shift

        # 8 surrounding position colours
        color_3x3 = state.board[ix_3x3, iy_3x3]
        pattern = pattern * 10 + color_3x3[0] + color_shift
        pattern = pattern * 10 + color_3x3[1] + color_shift
        pattern = pattern * 10 + color_3x3[2] + color_shift
        pattern = pattern * 10 + color_3x3[3] + color_shift
        pattern = pattern * 10 + color_3x3[4] + color_shift
        pattern = pattern * 10 + color_3x3[5] + color_shift
        pattern = pattern * 10 + color_3x3[6] + color_shift
        pattern = pattern * 10 + color_3x3[7] + color_shift

        # 8 surrounding position liberties
        liberty_3x3 = state.liberty_counts[ix_3x3, iy_3x3]
        pattern = pattern * 10 + min(max(liberty_3x3[0], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[1], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[2], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[3], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[4], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[5], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[6], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_3x3[7], 0), max_liberty_count)

        return pattern


def timeit():
    """ Testing performance
    """
    import random
    import sys
    import time
    import traceback
    import AlphaGo.go as go

    b_size = 19
    rf = RolloutFeature(b_size, f_pattern_3x3='./patterns/dict_non_response_3x3_max_4500.pat')
    
    cdef np.ndarray[DTYPE_t, ndim=2] F
    F = np.zeros((b_size**2, rf.n_feature), dtype=DTYPE)

    elapsed_s = list()
    num_error = 0
    for i in xrange(200):
        gs = go.GameState()

        empties = range(361)
        for j in xrange(200):
            move = random.choice(empties)
            empties.remove(move)
            x = int(move / 19)
            y = int(move % 19)
            try:
                s = time.time()
                gs.do_move((x, y))
                rf.update(gs, F)
                elapsed_s.append(time.time() - s)
            except go.IllegalMove:
                # sys.stderr.write("{}\n".format(traceback.format_exc()))
                num_error += 1
                continue

    print("Avg. {:.3f}us Err. {:d}".format(np.mean(elapsed_s)*1000*1000, num_error))
