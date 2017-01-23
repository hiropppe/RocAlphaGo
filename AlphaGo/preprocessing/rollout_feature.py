import numpy as np


class RolloutFeature(object):
    """
    """

    __NEIGHBOR8_CACHE = {}

    def __init__(self, state, pat3x3_file=None, pat12d_file=None):
        self.state = state
        self.size = state.size

        n_position = self.size ** 2
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
            self.n_3x3 = 0
            self.n_12d = 0

        n_feature = self.n_response + self.n_save_atari + self.n_neighbor + self.n_3x3 + self.n_12d
        self.feature = np.zeros((n_position, n_feature))

        self.idx_response = (0, self.n_response)
        self.idx_save_atari = (self.idx_response[1], self.idx_response[1] + self.n_save_atari)
        self.idx_neighbor = (self.idx_save_atari[1], self.idx_save_atari[1] + self.n_neighbor)
        self.idx_3x3 = (self.idx_neighbor[1], self.idx_neighbor[1] + self.n_3x3)
        self.idx_12d = (self.idx_3x3[1], self.idx_3x3[1] + self.n_12d)

        self._create_neighbor8_cache()
        self._prev_neighbors = None

    def _get_neighbor8(self, center):
        (x, y) = center
        neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                     (x,   y-1),           (x,   y+1),
                     (x+1, y-1), (x+1, y), (x+1, y+1)]
        xy = np.array([[nx*self.size + ny, self.idx_neighbor[0]+i]
                       for i, (nx, ny) in enumerate(neighbors) if self.state._on_board((nx, ny))])
        return (xy[:, 0], xy[:, 1])

    def _create_neighbor8_cache(self):
        for x in range(self.size):
            for y in range(self.size):
                RolloutFeature.__NEIGHBOR8_CACHE[(x, y)] = self._get_neighbor8((x, y))

    def update(self):
        if self.state.history:
            prev_move = self.state.history[-1]
        else:
            prev_move = None

        self.update_neighbors(prev_move)
        self.update_save_atari()

    def update_save_atari(self):
        self.feature[:, self.idx_save_atari[0]] = 0
        for (x, y) in self.state.last_liberty_cache[self.state.current_player]:
            if len(self.state.liberty_sets[x][y]) > 1:
                self.feature[x*self.size + y, self.idx_save_atari[0]] = 1
            else:
                for (nx, ny) in self.state._neighbors((x, y)):
                    if (self.state.board[nx, ny] == self.state.current_player
                       and self.state.liberty_counts[nx, ny] > 1):
                        self.feature[x*self.size + y, self.idx_save_atari[0]] = 1

    def update_neighbors(self, prev_move):
        # clear previous neighbors
        if self._prev_neighbors:
            self.feature[self._prev_neighbors[0], self._prev_neighbors[1]] = 0

        if prev_move:
            # set new neighbors
            (nx_ix, ny_ix) = RolloutFeature.__NEIGHBOR8_CACHE[prev_move]
            self.feature[nx_ix, ny_ix] = 1
            self._prev_neighbors = (nx_ix, ny_ix)

    def get_tensor():
        pass

    def get_sparse_tensor():
        pass
