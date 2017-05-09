# -*- coding: utf-8 -*-
#cython: boundscheck=False
"""
#cython: wraparound=False
"""
import numpy as np

cimport numpy as np

from libc.math cimport exp as cexp

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from AlphaGo.preprocessing import nakade

DTYPE = np.int

ctypedef np.float_t DTYPE_t
ctypedef np.float_t DTYPE_f

cdef int WHITE = -1
cdef int EMPTY = 0
cdef int BLACK = 1
cdef int OUT_OF_BOARD = 2

cdef enum:
    color_shift = 2
    max_liberty_count = 3


cdef class RolloutFeature(object):
    cdef readonly int bsize
    cdef readonly int n_position
    cdef readonly int n_features

    cdef readonly unordered_map[long long, int] pattern_3x3
    cdef pattern_d12
    # cdef readonly unordered_map[long long, int] pattern_d12

    cdef readonly int ix_response
    cdef readonly int ix_save_atari
    cdef readonly int ix_neighbor
    cdef readonly int ix_nakade
    cdef readonly int ix_3x3
    cdef readonly int ix_d12

    cdef int n_response
    cdef int n_save_atari
    cdef int n_neighbor
    cdef int n_nakade
    cdef int n_3x3
    cdef int n_d12

    cdef readonly unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR4_INDEX_CACHE
    cdef readonly unordered_map[int, pair[vector[int], vector[int]]] __NEIGHBOR8_INDEX_CACHE
    cdef readonly unordered_map[int, pair[vector[int], vector[int]]] __D12_INDEX_CACHE

    # neighbor feature index cache for each position. size = 3 [x_list, y_list, z_list]
    cdef unordered_map[int, vector[vector[int]]] __NEIGHBOR8_FEATURE_CACHE

    # previous neighbor feature index cache. size = 3 [x_list, y_list, z_list]
    cdef vector[vector[int]] _prev_neighbors

    cdef np.ndarray prev_board

    cdef readonly int hit_nakade
    cdef readonly int hit_3x3
    cdef readonly int hit_d12

    def __init__(self, bsize=19, f_pattern_3x3=None, f_pattern_d12=None):
        self.bsize = bsize
        self.n_position = self.bsize * self.bsize

        self.n_response = 1
        self.n_save_atari = 1
        self.n_neighbor = 8
        self.n_nakade = 24
        self.n_3x3 = 8*(3+4) 
        self.n_d12 = 12*(3+4)

        if f_pattern_3x3 or f_pattern_d12:
            import ast
            if f_pattern_3x3:
                with open(f_pattern_3x3) as f:
                    self.pattern_3x3 = ast.literal_eval(f.read())
                    self.n_3x3 = max(self.pattern_3x3.values()) + 1
            if f_pattern_d12:
                with open(f_pattern_d12) as f:
                    self.pattern_d12 = ast.literal_eval(f.read())
                    self.n_d12 = max(self.pattern_d12.values()) + 1

        self.n_features = (self.n_response + self.n_save_atari + self.n_neighbor +
                          self.n_nakade + self.n_3x3 + self.n_d12)

        self.ix_response = 0
        self.ix_save_atari = self.ix_response + self.n_response
        self.ix_neighbor = self.ix_save_atari + self.n_save_atari
        self.ix_nakade = self.ix_neighbor + self.n_neighbor
        self.ix_3x3 = self.ix_nakade + self.n_nakade
        self.ix_d12 = self.ix_3x3 + self.n_3x3

        self._create_neighbor8_cache()

        cdef vector[int] ix
        cdef vector[int] iy
        cdef vector[int] iz
        self._prev_neighbors.push_back(ix)
        self._prev_neighbors.push_back(iy)
        self._prev_neighbors.push_back(iz)

        self.prev_board = np.zeros((bsize, bsize), dtype=np.int)

        nakade.initialize_hash()

    def reset(self):
        self.prev_board = np.zeros((self.bsize, self.bsize), dtype=np.int)
        self.hit_nakade = 0
        self.hit_3x3 = 0
        self.hit_d12 = 0

    def _create_neighbor8_cache(self):
        for x in xrange(self.bsize):
            for y in xrange(self.bsize):
                position = x*self.bsize+y
                (fx, fy, fz) = self._get_neighbor8_feature_index((x, y))
                self.__NEIGHBOR8_FEATURE_CACHE[position].push_back(fx)
                self.__NEIGHBOR8_FEATURE_CACHE[position].push_back(fy)
                self.__NEIGHBOR8_FEATURE_CACHE[position].push_back(fz)
                # cache neighbor4
                (ix4, iy4) = self._get_neighbor4_index((x, y))
                self.__NEIGHBOR4_INDEX_CACHE[position].first = ix4
                self.__NEIGHBOR4_INDEX_CACHE[position].second = iy4
                # cache neighbor8
                (ix8, iy8) = self._get_neighbor8_index((x, y))
                self.__NEIGHBOR8_INDEX_CACHE[position].first = ix8
                self.__NEIGHBOR8_INDEX_CACHE[position].second = iy8
                # cache 12 point diamond
                (ix_d12, iy_d12) = self._get_d12_index((x, y))
                self.__D12_INDEX_CACHE[position].first = ix_d12
                self.__D12_INDEX_CACHE[position].second = iy_d12

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

    def _get_neighbor8_feature_index(self, center):
        """Returns nenighbor feature index ([np0, np1, .., np7], [ni0, ni1, .., ni7])
        """
        neighbors = self._get_neighbor8(center)
        xyz = np.array([[nx, ny, self.ix_neighbor+i]
                       for i, (nx, ny) in enumerate(neighbors) if self._on_board((nx, ny))])
        return (list(xyz[:, 0]), list(xyz[:, 1]), list(xyz[:, 2]))

    def _get_d12_index(self, center):
        """Returns 12 point diamond index pair ([r0, r1, .., r11], [c0, c1, .., c11])
        """
        neighbors = self._get_d12(center)
        xy = np.array(neighbors)
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

    def _get_d12(self, center):
        (x, y) = center
        return [                      (x-2, y),
                          (x-1, y-1), (x-1, y), (x-1, y+1),
                (x, y-2), (x,   y-1),           (x,   y+1), (x, y+2),
                          (x+1, y-1), (x+1, y), (x+1, y+1),
                                      (x+2, y)                      ]

    def _on_board(self, position):
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.bsize and y < self.bsize

    def update(self, state, np.ndarray[DTYPE_t, ndim=3] feature):
        """
        """
        cdef int px, py, pp

        if state.history and state.history[-1] is not None:
            pm = state.history[-1]
            if pm:
                (px, py) = pm
                pp = px*self.bsize+py
        else:
            pp = -1

        # Get updated positions by previous move
        updated_positions = self.get_updated_positions(state)

        self.update_neighbors(pp, feature)
        self.update_save_atari(state, feature)
        
        # Naive pattern feature
        self.update_naive_non_response_3x3(state, feature, updated_positions)
        self.update_naive_response_d12(state, feature)

        # Large one-hot feature. not work well and too slow
        # self.update_response_d12(state, feature)
        # self.update_non_response_3x3(state, feature)
        # self.update_nakade(state, feature)

    def update_naive_response_d12(self,
                                  state,
                                  np.ndarray[DTYPE_t, ndim=3] feature):
        """ Move matches 12-point diamond pattern near previous move
            length:
                color = 3(bit) * 12(position)
                liberty = 4(bit) * 12(position)
                #next move = 1(x-distance) + 1(y-distance)
            encoding:
                color: off=000 white=100 empty=010 black=001
                liberty: 0=1000 1=0100 2=0010 >2=0001
                #next move: 1=0 2=1
        """
        if not state.history or not state.history[-1]:
            return

        (px, py) = state.history[-1]
        if px < 2 or py < 2 or px >= self.bsize - 2 or py >= self.bsize - 2:
            return

        pp = px*self.bsize + py

        fx_vec = self.__D12_INDEX_CACHE[pp].first
        fy_vec = self.__D12_INDEX_CACHE[pp].second

        # convert to feature index
        # color: shift +1 (w=-1, e=0, b=1) -> (w=0, e=1, b=2)
        # liberty: clip liberty count 0,3 
        i_color = state.board[fx_vec, fy_vec] + 1 + self.ix_d12

        i_liberty_start = self.ix_d12 + 12 * 3
        i_liberty = state.liberty_counts[fx_vec, fy_vec]
        i_liberty = i_liberty.clip(0, 3)
        i_liberty = i_liberty + i_liberty_start 

        # each position has feature
        feature[i_color, fx_vec, fy_vec] = 1
        feature[i_liberty, fx_vec, fy_vec] = 1


    def update_naive_non_response_3x3(self, state,
                                      np.ndarray[DTYPE_t, ndim=3] feature,
                                      vector[int] updated_positions):
        """ Move matches 3 × 3 pattern around move
            length:
                color = 3(bit) * 8(position)
                liberty = 4(bit) * 8(position)
            encoding:
                color: off=000 white=100 empty=010 black=001
                liberty: 0=1000 1=0100 2=0010 >2=0001
        """
        # Update pattern around updated vertexes
        for up in updated_positions:
            feature_index = self.__NEIGHBOR8_INDEX_CACHE[up]
            fx_vec = feature_index.first
            fy_vec = feature_index.second

            for (fx, fy) in zip(fx_vec, fy_vec):
                # workaround. ignore position which is to close to the edge
                if fx < 1 or fy < 1 or fx >= self.bsize - 1 or fy >= self.bsize - 1:
                    continue

                fp = fx * self.bsize + fy
                nx_vec = self.__NEIGHBOR8_INDEX_CACHE[fp].first
                ny_vec = self.__NEIGHBOR8_INDEX_CACHE[fp].second

                # convert to feature index
                # color: shift +1 (w=-1, e=0, b=1) -> (w=0, e=1, b=2)
                # liberty: clip liberty count 0,3 
                i_color = state.board[nx_vec, ny_vec] + 1 + self.ix_3x3

                i_liberty_start = self.ix_3x3 + 8 * 3
                i_liberty = state.liberty_counts[nx_vec, ny_vec]
                i_liberty = i_liberty.clip(0, 3)
                i_liberty = i_liberty + i_liberty_start 

                for i_3x3 in range(8):
                    feature[i_3x3*3 + i_color[i_3x3], fx, fy] = 1
                    feature[i_3x3*3 + i_liberty[i_3x3], fx, fy] = 1


    def update_neighbors(self, prev_position, np.ndarray[DTYPE_t, ndim=3] feature):
        """ Move is 8-connected to previous move
        """
        # clear previous neighbors
        if not self._prev_neighbors[0].empty():
            prev_nx = self._prev_neighbors[0]
            prev_ny = self._prev_neighbors[1]
            prev_nz = self._prev_neighbors[2]
            feature[prev_nz, prev_nx, prev_ny] = 0

        if 0 <= prev_position:
            # set new neighbors
            prev = self.__NEIGHBOR8_FEATURE_CACHE[prev_position]
            nx = prev[0]
            ny = prev[1]
            nz = prev[2]
            feature[nz, nx, ny] = 1
            self._prev_neighbors[0] = nx
            self._prev_neighbors[1] = ny
            self._prev_neighbors[2] = nz

    def update_save_atari(self, state, np.ndarray[DTYPE_t, ndim=3] feature):
        """ Move saves stone(s) from capture
        """
        feature[self.ix_save_atari, :] = 0
        for (x, y) in state.last_liberty_cache[state.current_player]:
            if len(state.liberty_sets[x][y]) > 1:
                feature[self.ix_save_atari, x, y] = 1
            else:
                for (nx, ny) in state._neighbors((x, y)):
                    if (state.board[nx, ny] == state.current_player
                       and state.liberty_counts[nx, ny] > 1):
                        feature[self.ix_save_atari, x, y] = 1

    def update_nakade(self, state, np.ndarray[DTYPE_t, ndim=3] feature):
        """ Hash based large one-hot feature of nakade
            Not work well...
        """
        feature[self.ix_nakade:self.ix_3x3, :] = 0
        (pos, shape_id) = nakade.search_nakade(state)
        if pos != -1:
            self.hit_nakade += 1
            (x, y) = divmod(pos, self.bsize)
            feature[self.ix_nakade + shape_id, x, y] = 1

    def update_response_d12(self, state, np.ndarray[DTYPE_t, ndim=3] feature):
        """ Hash based large one-hot feature of 12-point diamond pattern
            Not work well...
        """
        #cdef unsigned long long pattern = 0
        #cdef unsigned long long pattern_by_location
        cdef vector[int] ix_d12, iy_d12
        cdef vector[int] color_d12, liberty_d12
        cdef int px, py, pp, dx, dy, fx, fy, fz

        if not state.history or not state.history[-1]:
            return

        (px, py) = state.history[-1]
        if px < 2 or py < 2 or px >= self.bsize - 2 or py >= self.bsize - 2:
            return
        pp = px*self.bsize + py

        d12_index = self.__D12_INDEX_CACHE[pp]
        ix_d12 = d12_index.first
        iy_d12 = d12_index.second
        moves = zip(ix_d12, iy_d12)

        color_d12 = state.board[ix_d12, iy_d12]
        pattern = long(color_d12[0]) + color_shift
        pattern = pattern * 10 + color_d12[1] + color_shift
        pattern = pattern * 10 + color_d12[2] + color_shift
        pattern = pattern * 10 + color_d12[3] + color_shift

        pattern = pattern * 10 + color_d12[4] + color_shift
        pattern = pattern * 10 + color_d12[5] + color_shift
        pattern = pattern * 10 + state.board[px, py] + color_shift
        pattern = pattern * 10 + color_d12[6] + color_shift
        pattern = pattern * 10 + color_d12[7] + color_shift

        pattern = pattern * 10 + color_d12[8] + color_shift
        pattern = pattern * 10 + color_d12[9] + color_shift
        pattern = pattern * 10 + color_d12[10] + color_shift

        pattern = pattern * 10 + color_d12[11] + color_shift

        liberty_d12 = state.liberty_counts[ix_d12, iy_d12]
        pattern = pattern * 10 + min(max(liberty_d12[0], 0), max_liberty_count)

        pattern = pattern * 10 + min(max(liberty_d12[1], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[2], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[3], 0), max_liberty_count)
        
        pattern = pattern * 10 + min(max(liberty_d12[4], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[5], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(state.liberty_counts[px, py], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[6], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[7], 0), max_liberty_count)
        
        pattern = pattern * 10 + min(max(liberty_d12[8], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[9], 0), max_liberty_count)
        pattern = pattern * 10 + min(max(liberty_d12[10], 0), max_liberty_count)
        
        pattern = pattern * 10 + min(max(liberty_d12[11], 0), max_liberty_count)

        for i in xrange(12):
            (fx, fy) = moves[i]
            dx = fx - px + 2
            dy = fy - py + 2

            pattern_by_location = pattern + (10 * dx + dy) * 10**26
            # if(self.pattern_d12.end() != self.pattern_d12.find(pattern_by_location)):
            if pattern_by_location in self.pattern_d12:
                self.hit_d12 += 1
                pattern_id = self.pattern_d12[pattern_by_location]
                fz = self.ix_d12 + pattern_id
                feature[fz, fx, fy] = 1 

    def update_non_response_3x3(self, state, np.ndarray[DTYPE_t, ndim=3] feature):
        """ Hash based large one-hot feature of 3x3 pattern
            Not work well...
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

            updated_positions.push_back(dx*self.bsize+dy)

            # group = state.group_sets[dx][dy]
            # if group:
            #     for (gx, gy) in group:
            #         updated_positions.push_back(gx*self.bsize+gy)

        return updated_positions

    def update_3x3_around(self, state, vector[int] updated_positions, np.ndarray[DTYPE_t, ndim=3] feature):
        """
        """
        cdef long long pattern_key
        cdef int pattern_id
        cdef int cp, fx, fy, fz
        
        # Update pattern around updated vertexes
        for cp in updated_positions:
            neighbor8_index = self.__NEIGHBOR8_INDEX_CACHE[cp]
            ix8 = neighbor8_index.first
            iy8 = neighbor8_index.second
            centers = zip(ix8, iy8)
            for j in xrange(8):
                (fx, fy) = centers[j]
                pattern_key = self.get_3x3_key(state, (fx, fy))
                if pattern_key >= 0:
                    if(self.pattern_3x3.end() != self.pattern_3x3.find(pattern_key)):
                        self.hit_3x3 += 1
                        pattern_id = self.pattern_3x3[pattern_key]
                        fz = self.ix_3x3 + pattern_id
                        feature[fz, fx, fy] = 1

    def get_3x3_key(self, state, center):
        """
        """
        cdef long long pattern = 0
        cdef int x, y, p
        cdef vector[int] ix_3x3, iy_3x3
        cdef vector[int] color_3x3, liberty_3x3

        (x, y) = center
        # position is to close to the edge
        if x < 1 or y < 1 or x >= self.bsize - 1 or y >= self.bsize - 1:
            return -1

        p = x * self.bsize + y
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


cdef int update_logits(int bsize, int n_features,
         np.ndarray[DTYPE_f, ndim=1] L,
         np.ndarray[DTYPE_f, ndim=2] X,
         np.ndarray[DTYPE_f, ndim=1] W,
         pair[vector[int], vector[int]] d12_cache,
         list elapsed_logits) except? -1:

    import time

    cdef int n = 12, p, i, x, y
    cdef double w_sum = 0
    cdef vector[int] ix_d12, iy_d12
    cdef np.ndarray[DTYPE_f, ndim=1] tmp

    ix_d12 = d12_cache.first
    iy_d12 = d12_cache.second

    s = time.time()

    for i in range(n):
        x = ix_d12[i]
        y = iy_d12[i]
        p = x*bsize+y
        """
        L[p] = W[X[p] == 1].sum()
        """

        """
        tmp = W[X[p] == 1]
        w_sum = 0
        for w in tmp:
            w_sum += w
        L[p] = w_sum
        """

        tmp = X[p]
        w_sum = 0
        for j in range(n_features):
            if tmp[j]:
                w_sum += W[j]
        L[p] = w_sum

    elapsed_logits.append(time.time() - s)

cdef softmax(np.ndarray[DTYPE_f, ndim=1] x,
             list elapsed_softmax):
    import time

    cdef double x_sum = 0, x_max = 0
    cdef int n = 361

    s = time.time()

    for i in range(361):
        if x_max < x[i]:
            x_max = x[i]
        x_sum += cexp(x[i])

    x = x - x_max
    y = np.exp(x) / x_sum
    # y = np.exp(x) / np.sum(np.exp(x))

    elapsed_softmax.append(time.time()-s)
    return y

cpdef timeit():
    """ Testing performance
    """
    import random
    import sys
    import time
    import traceback
    import AlphaGo.go as go

    cdef np.ndarray[DTYPE_f, ndim=3] F
    cdef np.ndarray[DTYPE_f, ndim=1] L, W

    bsize = 19
    rf = RolloutFeature(bsize)
    #rf = RolloutFeature(bsize,
    #                    f_pattern_3x3='./patterns/dict_non_response_3x3_max_4500.pat',
    #                    f_pattern_d12='./patterns/dict_response_12d_max_2300.pat')

    print("Board Size: {:d} ".format(bsize) +
          "Number of features: {:d} ".format(rf.n_features))

    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=rf.n_features)

    elapsed_feature = list()
    elapsed_move = list()
    elapsed_logits = list()
    elapsed_softmax = list()
    hits_nakade = list()
    hits_d12 = list()
    hits_3x3 = list()
    num_error = 0
    for i in xrange(100):
        gs = go.GameState()
        rf.reset() 
        F = np.zeros((rf.n_features, bsize, bsize), dtype=np.float)
        L = np.zeros((bsize**2), dtype=np.float)
        empties = range(361)
        for j in xrange(200):
            move = random.choice(empties)
            empties.remove(move)
            (x, y) = divmod(move, bsize)
            try:
                s = time.time()
                gs.do_move((x, y))
                elapsed_move.append(time.time() - s)
                s = time.time()
                rf.update(gs, F)
                elapsed_feature.append(time.time() - s)
            except go.IllegalMove:
                # sys.stderr.write("{}\n".format(traceback.format_exc()))
                num_error += 1
                continue

            X = F.transpose(1, 2, 0).reshape(bsize**2, rf.n_features)
            if j == 0:
                L = X.dot(W)
            else:
                if not (x < 2 or y < 2 or x >= bsize - 2 or y >= bsize - 2):
                    d12_cache = rf.__D12_INDEX_CACHE[move]
                    update_logits(bsize, rf.n_features, L, X, W, d12_cache, elapsed_logits)
                    softmax(L, elapsed_softmax)

            """
            dot = X.dot(W)
            tmp_dot = np.trunc(dot * 10 ** 3)/10 ** 3
            tmp_L = np.trunc(L * 10 ** 3)/10 ** 3
            if not np.all(tmp_dot == tmp_L):
                print 'Loop position: ({:d}, {:d})'.format(i, j)
                print 'Move ({:d}, {:d})'.format(x, y)
                print 'dot == L', dot == L
                print 'np.where(dot != L)', np.where(dot != L)
                print 'dot[np.where(dot != L)]', dot[np.where(dot != L)]
                print 'L[np.where(dot != L)]', L[np.where(dot != L)]
                print 'dot', dot
                print 'L', L

            assert np.all(X.dot(W) == L), 'assertion error at pos={:d}'.format(move)
            """

        hits_nakade.append(rf.hit_nakade)
        hits_3x3.append(rf.hit_3x3)
        hits_d12.append(rf.hit_d12)

    print("Avg Update Feature. {:.3f} us. ".format(np.mean(elapsed_feature)*1000*1000) +
          "Avg Move. {:.3f} us. ".format(np.mean(elapsed_move)*1000*1000) +
          "Avg Logits. {:.3f} us. ".format(np.mean(elapsed_logits)*1000*1000) +
          "Avg Softmax. {:.3f} us. ".format(np.mean(elapsed_softmax)*1000*1000) +
          "nakade hits: {:.3f}. ".format(np.mean(hits_nakade)) +
          "3x3 hits: {:.3f}. ".format(np.mean(hits_3x3)) +
          "12d hits: {:.3f}. ".format(np.mean(hits_d12)) +
          "Err: {:d}".format(num_error))
