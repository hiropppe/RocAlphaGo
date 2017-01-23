import numpy as np

import AlphaGo.go as go
import pattern as ptn

from scipy.sparse import lil_matrix, csr_matrix, hstack


def get_response(state):
    feature = lil_matrix((state.size**2, 1))
    if state.history and state.history[-1]:
        response = ptn.get_diamond_index(state.history[-1],
                                         state.size,
                                         reshape=False)
        feature[response] = 1
    return feature.tocsr()


def get_save_atari(state):
    """ Move saves stone(s) from capture
    """
    feature = lil_matrix((state.size**2, 1))
    for (x, y) in state.get_legal_moves():
        for neighbor_group in state.get_groups_around((x, y)):
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                lib_count = state.liberty_counts[gx][gy]
                if lib_count == 1:
                    (lx, ly) = next(iter(state.liberty_sets[gx][gy]))
                    lib_set_after = state.liberty_sets[lx][ly]
                    if len(lib_set_after) >= 2:
                        feature[lx*state.size+ly, 0] = 1
    return feature.tocsr()


def get_neighbour(state):
    """ Move is 8-connected to previous move
    """
    feature = lil_matrix((state.size**2, 8))
    if state.history and state.history[-1]:
        px, py = state.history[-1]
        center = px*state.size + py
        for i, distance in enumerate([-(state.size+1), -state.size, -(state.size-1),
                                      -1, 1,
                                      state.size-1, state.size, state.size+1]):
            x = center + distance
            if 0 <= x and x < state.size**2:
                feature[x, i] = 1
    return feature.tocsr()


def get_non_response_pattern(state, pat_dict):
    """ Move matches 3 x 3 pattern around move
    """
    feature_len = len(pat_dict)
    feature = lil_matrix((state.size**2, feature_len))
    for (x, y) in state.get_legal_moves():
        if state.current_player == go.WHITE:
            reverse = True
        else:
            reverse = False
        if not state.history or not state.history[-1] or not ptn.is_in_manhattan(state.history[-1], x, y):
            key = ptn.get_3x3_color_value(state, (x, y), symmetric=False, reverse=reverse)
            try:
                idx = pat_dict[key]
                feature[x*state.size+y, idx] = 1
            except KeyError:
                pass
    return feature.tocsr()


def get_response_pattern(state, pat_dict):
    """ Move matches 12-point diamond pattern near previous move
    """
    feature_len = len(pat_dict)
    feature = lil_matrix((state.size**2, feature_len))
    if state.history and state.history[-1]:
        if state.current_player == go.BLACK:
            reverse = True
        else:
            reverse = False

        key = ptn.get_diamond_color_value(state, state.history[-1], symmetric=True, reverse=reverse)
        try:
            idx = pat_dict[key]
            for (x, y) in state.get_legal_moves():
                if ptn.is_in_manhattan(state.history[-1], x, y):
                    feature[x*state.size+y, idx] = 1
        except KeyError:
            pass
    return feature.tocsr()


FEATURES = {
    "ones": {
        "size": 1,
        "function": lambda s: csr_matrix(np.ones((s.size**2, 1)))
    },
    "response": {
        "size": 1,
        "function": get_response
    },
    "save_atari": {
        "size": 1,
        "function": get_save_atari
    },
    "neighbour": {
        "size": 8,
        "function": get_neighbour
    },
    "response_pattern": {
        "function": get_response_pattern
    },
    "non_response_pattern": {
        "function": get_non_response_pattern
    },
}

DEFAULT_FEATURES = [
    "ones", "response", "save_atari", "neighbour", "response_pattern", "non_response_pattern"
]


class RolloutPreprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, pat_3x3_file=None, pat_dia_file=None, feature_list=DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """
        try:
            import cPickle as pkl
        except:
            import pickle as pkl

        if pat_3x3_file:
            self.pat_3x3_dict = pkl.load(open(pat_3x3_file, "r"))
        if pat_dia_file:
            self.pat_dia_dict = pkl.load(open(pat_dia_file, "r"))

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                if feat == "non_response_pattern":
                    f = FEATURES[feat]["function"]
                    self.processors[i] = lambda s: f(s, self.pat_3x3_dict)
                    self.output_dim += len(self.pat_3x3_dict)
                elif feat == "response_pattern":
                    f = FEATURES[feat]["function"]
                    self.processors[i] = lambda s: f(s, self.pat_dia_dict)
                    self.output_dim += len(self.pat_dia_dict)
                else:
                    self.processors[i] = FEATURES[feat]["function"]
                    self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state) for proc in self.processors]
        return hstack(feat_tensors).tocsr()
