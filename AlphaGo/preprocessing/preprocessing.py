import numpy as np
import AlphaGo.go as go
import keras.backend as K

import pattern as ptn

# This file is used anywhere that neural net features are used; setting the keras dimension ordering
# here makes it universal to the project.
K.set_image_dim_ordering('th')

##
# individual feature functions (state --> tensor) begin here
##


def get_board(state):
    """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    planes = np.zeros((3, state.size, state.size))
    planes[0, :, :] = state.board == state.current_player  # own stone
    planes[1, :, :] = state.board == -state.current_player  # opponent stone
    planes[2, :, :] = state.board == go.EMPTY  # empty space
    return planes


def get_turns_since(state, maximum=8):
    """A feature encoding the age of the stone at each location up to 'maximum'

    Note:
    - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
    - EMPTY locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size))
    for x in range(state.size):
        for y in range(state.size):
            if state.stone_ages[x][y] >= 0:
                planes[min(state.stone_ages[x][y], maximum - 1), x, y] = 1
    return planes


def get_liberties(state, maximum=8):
    """A feature encoding the number of liberties of the group connected to the stone at
    each location

    Note:
    - there is no zero-liberties plane; the 0th plane indicates groups in atari
    - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
    - EMPTY locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size))
    for i in range(maximum):
        # single liberties in plane zero (groups won't have zero), double
        # liberties in plane one, etc
        planes[i, state.liberty_counts == i + 1] = 1
    # the "maximum-or-more" case on the backmost plane
    planes[maximum - 1, state.liberty_counts >= maximum] = 1
    return planes


def get_capture_size(state, maximum=8):
    """A feature encoding the number of opponent stones that would be captured by
    playing at each location, up to 'maximum'

    Note:
    - we currently *do* treat the 0th plane as "capturing zero stones"
    - the [maximum-1] plane is used for any capturable group of size
      greater than or equal to maximum-1
    - the 0th plane is used for legal moves that would not result in capture
    - illegal move locations are all-zero features

    """
    planes = np.zeros((maximum, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        # multiple disconnected groups may be captured. hence we loop over
        # groups and count sizes if captured.
        n_captured = 0
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is opponent stones and they have
            # one liberty, it must be (x,y) and we are capturing them
            # (note suicide and ko are not an issue because they are not
            # legal moves)
            (gx, gy) = next(iter(neighbor_group))
            if (state.liberty_counts[gx][gy] == 1) and \
               (state.board[gx, gy] != state.current_player):
                n_captured += len(state.group_sets[gx][gy])
        planes[min(n_captured, maximum - 1), x, y] = 1
    return planes


def get_self_atari_size(state, maximum=8):
    """A feature encoding the size of the own-stone group that is put into atari by
    playing at a location

    """
    planes = np.zeros((maximum, state.size, state.size))

    for (x, y) in state.get_legal_moves():
        # make a copy of the liberty/group sets at (x,y) so we can manipulate them
        lib_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x, y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                lib_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        if (x, y) in lib_set_after:
            lib_set_after.remove((x, y))
        # check if this move resulted in atari
        if len(lib_set_after) == 1:
            group_size = len(group_set_after)
            # 0th plane used for size=1, so group_size-1 is the index
            planes[min(group_size - 1, maximum - 1), x, y] = 1
    return planes


def get_liberties_after(state, maximum=8):
    """A feature encoding what the number of liberties *would be* of the group connected to
    the stone *if* played at a location

    Note:
    - there is no zero-liberties plane; the 0th plane indicates groups in atari
    - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
    - illegal move locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size))
    # note - left as all zeros if not a legal move
    for (x, y) in state.get_legal_moves():
        # make a copy of the set of liberties at (x,y) so we can add to it
        lib_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x, y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group and
            # therefore add in all that group's liberties
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                lib_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        # (x,y) itself may have made its way back in, but shouldn't count
        # since it's clearly not a liberty after playing there
        if (x, y) in lib_set_after:
            lib_set_after.remove((x, y))
        planes[min(maximum - 1, len(lib_set_after) - 1), x, y] = 1
    return planes


def get_ladder_capture(state):
    """A feature wrapping GameState.is_ladder_capture().
    """
    feature = np.zeros((1, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = state.is_ladder_capture((x, y))
    return feature


def get_ladder_escape(state):
    """A feature wrapping GameState.is_ladder_escape().
    """
    feature = np.zeros((1, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = state.is_ladder_escape((x, y))
    return feature


def get_sensibleness(state):
    """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
    """
    feature = np.zeros((1, state.size, state.size))
    for (x, y) in state.get_legal_moves(include_eyes=False):
        feature[0, x, y] = 1
    return feature


def get_legal(state):
    """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
    """
    feature = np.zeros((1, state.size, state.size))
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = 1
    return feature


def get_response(state):
    feature = np.zeros((state.size**2, 1))
    if state.history and state.history[-1]:
        for (x, y) in state.get_legal_moves():
            if ptn.is_in_manhattan(state.history[-1], x, y):
                feature[x*state.size+y] = 1
    return feature


def get_save_atari(state):
    """ Move saves stone(s) from capture
    """
    feature = np.zeros((state.size**2, 1))
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
    return feature


def get_neighbour(state):
    """ Move is 8-connected to previous move
    """
    feature = np.zeros((state.size**2, 8))
    if state.history and state.history[-1]:
        px, py = state.history[-1]
        for (x, y) in state.get_legal_moves():
            center = px*state.size + py
            move = x*state.size + y

            i = x*state.size + y
            if move == center - state.size - 1:
                feature[i, 0] = 1
            elif move == center - state.size:
                feature[i, 1] = 1
            elif move == center - state.size + 1:
                feature[i, 2] = 1
            elif move == center - 1:
                feature[i, 3] = 1
            elif move == center + 1:
                feature[i, 4] = 1
            elif move == center + state.size - 1:
                feature[i, 5] = 1
            elif move == center + state.size:
                feature[i, 6] = 1
            elif move == center + state.size + 1:
                feature[i, 7] = 1
            else:
                pass
    return feature


def get_response_pattern(state, pat_dict):
    """ Move matches 12-point diamond pattern near previous move
    """
    feature_len = len(bin(len(pat_dict)))-2
    feature = np.zeros((state.size**2, feature_len))
    if state.history and state.history[-1]:
        if state.current_player == go.BLACK:
            reverse = True
        else:
            reverse = False
        key = ptn.get_diamond_value(state, state.history[-1], symmetric=True, reverse=reverse)
        try:
            idx = pat_dict[key]
            for (x, y) in state.get_legal_moves():
                if ptn.is_in_manhattan(state.history[-1], x, y):
                    feature[x*state.size+y] = np.array([b for b in bin(idx)[2:].rjust(feature_len, '0')],
                                                       dtype=np.byte)
                    #feature[x*state.size+y, idx] = 1
        except KeyError:
            pass
    return feature


def get_non_response_pattern(state, pat_dict):
    """ Move matches 3 x 3 pattern around move
    """
    feature_len = len(bin(len(pat_dict)))-2
    feature = np.zeros((state.size**2, feature_len))
    for (x, y) in state.get_legal_moves():
        if state.current_player == go.WHITE:
            reverse = True
        else:
            reverse = False
        if not ptn.is_in_manhattan(state.history[-1], x, y):
            key = ptn.get_3x3_value(state, (x, y), symmetric=False, reverse=reverse)
            try:
                idx = pat_dict[key]
                feature[x*state.size+y] = np.array([b for b in bin(idx)[2:].rjust(feature_len, '0')],
                                                   dtype=np.byte)
                #feature[x*state.size+y, idx] = 1
            except KeyError:
                pass
    return feature


def get_response_pattern2(state):
    """ Move matches 12-point diamond pattern near previous move
    """
    feature = np.zeros((state.size**2, 4*12))
    if state.history and state.history[-1]:
        if state.current_player == go.BLACK:
            reverse = True
        else:
            reverse = False
        pat = ptn.get_diamond_color_pattern(state, state.history[-1], symmetric=True, reverse=reverse)
        for (x, y) in state.get_legal_moves():
            if ptn.is_in_manhattan(state.history[-1], x, y):
                feature[x*state.size+y] = pat
    return feature


def get_non_response_pattern2(state):
    """ Move matches 3 x 3 pattern around move
    """
    feature = np.zeros((state.size**2, 4*8))
    for (x, y) in state.get_legal_moves():
        if state.current_player == go.WHITE:
            reverse = True
        else:
            reverse = False
        if not state.history or not state.history[-1] or not ptn.is_in_manhattan(state.history[-1], x, y):
            pat = ptn.get_3x3_color_pattern(state, (x, y), symmetric=False, reverse=reverse)
            feature[x*state.size+y] = pat
    return feature


def get_response_pattern3(state):
    """ Move matches 12-point diamond pattern near previous move
    """
    feature = np.zeros((state.size**2, 8*12))
    if state.history and state.history[-1]:
        if state.current_player == go.BLACK:
            reverse = True
        else:
            reverse = False
        pat = ptn.get_diamond_pattern(state, state.history[-1], symmetric=True, reverse=reverse)
        for (x, y) in state.get_legal_moves():
            if ptn.is_in_manhattan(state.history[-1], x, y):
                feature[x*state.size+y] = pat
    return feature


def get_non_response_pattern3(state):
    """ Move matches 3 x 3 pattern around move
    """
    feature = np.zeros((state.size**2, 8*8))
    for (x, y) in state.get_legal_moves():
        if state.current_player == go.WHITE:
            reverse = True
        else:
            reverse = False
        if not state.history or not state.history[-1] or not ptn.is_in_manhattan(state.history[-1], x, y):
            pat = ptn.get_3x3_pattern(state, (x, y), symmetric=False, reverse=reverse)
            feature[x*state.size+y] = pat
    return feature


def get_distance(state):
    """ Manhattan distance to previous two moves
    """
    cap = state.size-2
    pattern = np.zeros((state.size**2, 2*cap))
    if 1 < len(state.history):
        prev2, prev1 = state.history[-2], state.history[-1]
        for (x, y) in state.get_legal_moves():
            i = y*state.size + x
            d = min(get_mdist(prev2, prev1, cap) + get_mdist(prev1, (x, y), cap), 2*cap)
            pattern[i, d-1] = 1
    return pattern


def get_mdist(p1, p2, cap):
    dx = min(abs(p1[0] - p2[0]), cap)
    dy = min(abs(p1[1] - p2[1]), cap)
    return dx + dy


# named features and their sizes are defined here
FEATURES = {
    "board": {
        "size": 3,
        "function": get_board
    },
    "ones": {
        "size": 1,
        "function": lambda state: np.ones((1, state.size, state.size))
    },
    "turns_since": {
        "size": 8,
        "function": get_turns_since
    },
    "liberties": {
        "size": 8,
        "function": get_liberties
    },
    "capture_size": {
        "size": 8,
        "function": get_capture_size
    },
    "self_atari_size": {
        "size": 8,
        "function": get_self_atari_size
    },
    "liberties_after": {
        "size": 8,
        "function": get_liberties_after
    },
    "ladder_capture": {
        "size": 1,
        "function": get_ladder_capture
    },
    "ladder_escape": {
        "size": 1,
        "function": get_ladder_escape
    },
    "sensibleness": {
        "size": 1,
        "function": get_sensibleness
    },
    "zeros": {
        "size": 1,
        "function": lambda state: np.zeros((1, state.size, state.size))
    },
    "legal": {
        "size": 1,
        "function": get_legal
    },
    "color": {
        "size": 1,
        "function": lambda state: np.ones((1, state.size, state.size)) *
        (state.current_player == go.BLACK)
    },
    # rollout
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
    "response_pattern2": {
        "size": 48,
        "function": get_response_pattern2
    },
    "non_response_pattern2": {
        "size": 32,
        "function": get_non_response_pattern2
    },
    "response_pattern3": {
        "size": 96,
        "function": get_response_pattern3
    },
    "non_response_pattern3": {
        "size": 64,
        "function": get_non_response_pattern3
    },
    # tree
    "distance": {
        "size": 34,
        "function": get_distance
    }
}

DEFAULT_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size",
    "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
    "sensibleness", "zeros"]


ROLLOUT_FEATURES = [
    "response", "neighbour"
]

TREE_FEATURES = [
    "response", "neighbour", "distance"
]


class Preprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list=DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state) for proc in self.processors]

        # concatenate along feature dimension then add in a singleton 'batch' dimension
        f, s = self.output_dim, state.size
        return np.concatenate(feat_tensors).reshape((1, f, s, s))


class RolloutPreprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list=ROLLOUT_FEATURES, pat_3x3_file=None, pat_dia_file=None):
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
                    self.output_dim += len(bin(len(self.pat_3x3_dict)))-2
                elif feat == "response_pattern":
                    f = FEATURES[feat]["function"]
                    self.processors[i] = lambda s: f(s, self.pat_dia_dict)
                    self.output_dim += len(bin(len(self.pat_dia_dict)))-2
                else:
                    self.processors[i] = FEATURES[feat]["function"]
                    self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state) for proc in self.processors]

        # concatenate along feature dimension then add in a singleton 'batch' dimension
        f, s = self.output_dim, state.size**2
        return np.concatenate(feat_tensors, axis=1).reshape((1, s, f))
