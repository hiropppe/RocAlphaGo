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

    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_FEATURE_CACHE
    cdef unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_INDEX_CACHE

    cdef pair[vector[int], vector[int]] _prev_neighbors

    cdef unordered_map[long long, int] pattern_3x3
    cdef unordered_map[long long, int] pattern_12d

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

    def _create_neighbor8_cache(self):
        for x in range(self.b_size):
            for y in range(self.b_size):
                position = x*self.b_size+y
                (fx, fy) = self._get_neighbor8_feature((x, y))
                self.__NEIGHBOR8_FEATURE_CACHE[position].first = fx
                self.__NEIGHBOR8_FEATURE_CACHE[position].second = fy
                # ignoring outermost
                if not (x < 1 or y < 1 or x >= self.b_size - 1 or y >= self.b_size - 1):
                    (ix, iy) = self._get_neighbor8_index((x, y))
                    self.__NEIGHBOR8_INDEX_CACHE[position].first = ix
                    self.__NEIGHBOR8_INDEX_CACHE[position].second = iy

    def _get_neighbor8_index(self, center):
        neighbors = self._get_neighbor8(center)
        xy = np.array(neighbors)
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _get_neighbor8_feature(self, center):
        neighbors = self._get_neighbor8(center)
        xy = np.array([[nx*self.b_size + ny, self.ix_neighbor+i]
                       for i, (nx, ny) in enumerate(neighbors) if self._on_board((nx, ny))])
        return (list(xy[:, 0]), list(xy[:, 1]))

    def _get_neighbor8(self, center):
        (x, y) = center
        return [(x-1, y-1), (x-1, y), (x-1, y+1),
                (x,   y-1),           (x,   y+1),
                (x+1, y-1), (x+1, y), (x+1, y+1)]

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
        self.update_non_response_3x3(state, feature)

        self.prev_board = state.board.copy()

    def update_non_response_3x3(self, state, np.ndarray[DTYPE_t, ndim=2] feature):
        cdef long long pattern_key
        cdef int xy, pattern_idx, fx, fy, cx, cy
        cdef int vlen, clen

        if not state.history or not state.history[-1]:
            return

        if len(state.history) == 1:
            updated_vertexes = [state.history[-1]]
        else:
            updated_vertexes = np.where(state.board-self.prev_board != 0) 
            updated_vertexes = zip(updated_vertexes[0], updated_vertexes[1])

        vlen = len(updated_vertexes)
        for i in range(vlen):
            (cx, cy) = updated_vertexes[i]
            if cx < 1 or cy < 1 or cx >= self.b_size - 1 or cy >= self.b_size - 1:
                continue

            xy = cx*self.b_size + cy
            ix_3x3 = self.__NEIGHBOR8_INDEX_CACHE[xy].first
            iy_3x3 = self.__NEIGHBOR8_INDEX_CACHE[xy].second

            centers = zip(ix_3x3, iy_3x3)
            clen = len(centers)
            for j in range(clen):
                (x, y) = centers[j]
                #if state.is_legal((x, y)):
                pattern_key = self.get_3x3_hash(state, (x, y))
                pattern_idx = self.pattern_3x3[pattern_key]
                fx = x*self.b_size+y
                fy = self.ix_3x3 + pattern_idx
                if(self.pattern_3x3.end() != self.pattern_3x3.find(pattern_key)):
                    feature[fx, fy] = 1

    def update_neighbors(self, prev_position, np.ndarray[DTYPE_t, ndim=2] feature):
        # clear previous neighbors
        if not self._prev_neighbors.first.empty():
            prev_nx = self._prev_neighbors.first
            prev_ny = self._prev_neighbors.second
            feature[prev_nx, prev_ny] = 0

        if 0 < prev_position:
            # set new neighbors
            prev = self.__NEIGHBOR8_FEATURE_CACHE[prev_position]
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


    def get_3x3_hash(self, state, center):
        cdef long long pattern_hash
        cdef vector[int] ix_3x3, iy_3x3, ixy_3x3
        cdef vector[int] color_3x3, liberty_3x3
        cdef int x, y, xy 

        (x, y) = center
        # position is to close to the edge
        if x < 1 or y < 1 or x >= self.b_size - 1 or y >= self.b_size - 1:
            return -1

        # active player colour
        pattern_hash = 2L
        pattern_hash += long(state.current_player)
        pattern_hash *= 10L

        xy = x * self.b_size + y
        ix_3x3 = self.__NEIGHBOR8_INDEX_CACHE[xy].first
        iy_3x3 = self.__NEIGHBOR8_INDEX_CACHE[xy].second
        # 8 surrounding position colours
        color_3x3 = state.board[ix_3x3, iy_3x3]
        for color in color_3x3:
            pattern_hash += color + 2L
            pattern_hash *= 10L
        # 8 surrounding position liberties
        liberty_3x3 = state.liberty_counts[ix_3x3, iy_3x3]
        for liberty in liberty_3x3:
            pattern_hash += min(liberty, 3)
            pattern_hash *= 10

        """
        ixy_3x3 = self.__NEIGHBOR8_FEATURE_CACHE[z].first
        for xy in ixy_3x3:
            # 8 surrounding position colours
            nx = xy / self.b_size
            ny = xy % self.b_size
            pattern_hash += state.board[nx, ny] + 2L
            pattern_hash *= 10L
            # 8 surrounding position liberties
            pattern_hash += min(state.liberty_counts[nx, ny], 3 )
            pattern_hash *= 10
        """

        return pattern_hash / 10


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
