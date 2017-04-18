# -*- coding: utf-8 -*-
#cython: boundscheck=False

import numpy as np

import sys

from Queue import Queue

cimport numpy as np

# from libcpp.algorithm cimport sort as stdsort
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort as stdsort

from cpprand cimport uniform_int_distribution, mt19937_64, random_device

DTYPE = np.int

ctypedef np.int_t DTYPE_t


cdef int BOARD_SIZE = 19
cdef int BOARD_MAX = BOARD_SIZE ** 2

cdef int board_size = BOARD_SIZE

cdef enum:
    HASH_PASS = 0
    HASH_BLACK = 1
    HASH_WHITE = 2
    HASH_KO = 3

cdef enum:
    NAKADE_3 = 6
    NAKADE_4 = 5
    NAKADE_5 = 9
    NAKADE_6 = 4

cdef unsigned long long[361][4] hash_bit
cdef unsigned long long[361] shape_bit

cdef int[4] NAKADE_PATTERNS
NAKADE_PATTERNS[0] = NAKADE_3
NAKADE_PATTERNS[1] = NAKADE_4
NAKADE_PATTERNS[2] = NAKADE_5
NAKADE_PATTERNS[3] = NAKADE_6

cdef int NAKADE_MAX_SIZE = NAKADE_5

cdef unsigned long long[4][9] nakade_hash
cdef int[4][9] nakade_pos

cdef int start = BOARD_MAX / 2 

cdef int[6][3] nakade3  # nakade3 pat
cdef int[5][4] nakade4  # nakade4 pat
cdef int[9][5] nakade5  # nakade5 pat
cdef int[4][6] nakade6  # nakade6 pat

ctypedef uniform_int_distribution[long long] int_dist

cdef int_dist *randint = new int_dist(0, sys.maxint)
cdef random_device *rand_gen = new random_device()
cdef int seed = rand_gen[0]()
cdef mt19937_64 *engine = new mt19937_64(seed)

cdef mt():
    return randint[0](engine[0])


cdef NORTH(pos):
    return pos - BOARD_SIZE if pos >= BOARD_SIZE else -1


cdef SOUTH(pos):
    return pos + BOARD_SIZE if pos + BOARD_SIZE < BOARD_MAX else -1


cdef WEST(pos):
    return pos - 1 if pos % BOARD_SIZE != 0 else -1


cdef EAST(pos):
    return pos + 1 if (pos + 1) % BOARD_SIZE != 0 else -1


cdef set_neighbor4(int[4] neighbor4, int center):
    neighbor4[0] = NORTH(center)
    neighbor4[1] = WEST(center)
    neighbor4[2] = EAST(center)
    neighbor4[3] = SOUTH(center)


cdef onboard(int x, int y):
    return 0 <= x and x < 19 and 0 <= y and y < 19


def search_nakade(state):
    cdef int last_color
    cdef int last_pos
    cdef int[4] neighbor4

    if not state.history:
        return

    last_move = state.history[-1]
    last_pos = last_move[0]*BOARD_SIZE+last_move[1]
    last_color = state.board[last_move[0], last_move[1]]
    print 'last', last_move, last_color
    set_neighbor4(neighbor4, last_pos)

    for i in xrange(4):
        if neighbor4[i] != -1:
            (ny, nx) = divmod(neighbor4[i], BOARD_SIZE)
            print 'Search', ny, nx
            nakade_pos = find_nakade_pos(state, neighbor4[i], last_color)
            if nakade_pos > -1:
                print 'Vital vertex of nakade', divmod(nakade_pos, 19)
                return nakade_pos


def is_nakade_self_atari(state, pos, color):
    cdef np.ndarray nakade = np.zeros((10,), dtype=np.int)
    cdef int nakade_num = 0
    cdef unsigned long long hash = 0
    cdef int[4] neighbor4
    cdef int gx, gy

    set_neighbor4(neighbor4, pos)
    north = divmod(neighbor4[0], BOARD_SIZE) if neighbor4[0] != -1 else None
    west  = divmod(neighbor4[1], BOARD_SIZE) if neighbor4[1] != -1 else None
    east  = divmod(neighbor4[2], BOARD_SIZE) if neighbor4[2] != -1 else None
    south = divmod(neighbor4[3], BOARD_SIZE) if neighbor4[3] != -1 else None

    empty_group = set()
    
    north_group = state.get_group(north) if north and state.board[north[0], north[1]] == color else empty_group
    west_group  = state.get_group(west)  if west  and state.board[west[0],  west[1]]  == color else empty_group
    east_group  = state.get_group(east)  if east  and state.board[east[0],  east[1]]  == color else empty_group
    south_group = state.get_group(south) if south and state.board[south[0], south[1]] == color else empty_group

    merged_group = south_group | west_group | north_group | east_group

    for i, (gy, gx) in enumerate(merged_group):
        nakade[i] = gy * BOARD_SIZE + gx
        nakade_num += 1

    nakade[nakade_num] = pos
    nakade_num += 1

    print 'nakade_num', nakade_num
    if nakade_num > 6:
        print 'nakade_num exceed threshold', nakade_num 
        return -1

    nakade.sort()
    nakade = nakade[-nakade_num:]
    print 'sorted nakade shape', nakade

    # centering pattern
    reviser = start - nakade[0]
    for i in xrange(nakade_num):
        print 'reivised shape pos', (nakade[i] + reviser)
        hash ^= shape_bit[nakade[i] + reviser]

    if nakade_num in (3, 4, 5, 6):
        for i in xrange(NAKADE_PATTERNS[nakade_num - 3]):
            print i, nakade_hash[nakade_num - 3][i], hash
            if nakade_hash[nakade_num - 3][i] == hash:
                return True

    return False


def find_nakade_pos(state, pos, color):
    cdef int size = 0
    cdef bool[361] flag
    # cdef int[10] nakade = [0]*10
    cdef np.ndarray nakade = np.zeros((10,), dtype=np.int)
    cdef int nakade_num = 0
    cdef unsigned long long hash = 0
    cdef int[4] neighbor4
    cdef int nx, ny, npos

    nakadebug = list()

    nakade_queue = Queue()
    nakade_queue.put(pos)

    flag = [False]*361
    flag[pos] = True

    board = state.board
    print board
    while not nakade_queue.empty():
        current_pos = nakade_queue.get()
        nakade[nakade_num] = current_pos
        nakade_num += 1
        nakadebug.append(divmod(current_pos, 19))
        print nakadebug

        if size > 5:
            print 'exit', size
            return -1

        set_neighbor4(neighbor4, current_pos)

        for i in xrange(4):
            if neighbor4[i] != -1:
                npos = neighbor4[i]
                (ny, nx) = divmod(neighbor4[i], BOARD_SIZE)
                nc = board[ny, nx]
                print (ny, nx), nc, flag[npos], color, (nc == 0 or nc != color)
                if not flag[npos] and (nc == 0 or nc != color):
                    # print 'append', flag[np], neighbor4[i]
                    nakade_queue.put(npos)
                    flag[npos] = True
                    size += 1

    print 'nakade_num', nakade_num
    if nakade_num > 6:
        print 'nakade_num exceed threshold', size
        return -1

    nakade.sort()
    nakade = nakade[-nakade_num:]
    print 'sorted nakade shape', nakade

    # centering pattern
    reviser = start - nakade[0]
    for i in xrange(nakade_num):
        print 'reivised shape pos', (nakade[i] + reviser)
        hash ^= shape_bit[nakade[i] + reviser]

    if nakade_num in (3, 4, 5, 6):
        for i in xrange(NAKADE_PATTERNS[nakade_num - 3]):
            print i, nakade_hash[nakade_num - 3][i], hash
            if nakade_hash[nakade_num - 3][i] == hash:
                return nakade[0] + nakade_pos[nakade_num - 3][i]

    return -1
    

def initialize_hash():

    for i in xrange(BOARD_MAX):
        hash_bit[i][HASH_PASS] = mt() 
        hash_bit[i][HASH_BLACK] = mt()
        hash_bit[i][HASH_WHITE] = mt()
        hash_bit[i][HASH_KO] = mt()
        shape_bit[i] = mt()

    nakade3[0][0] = 0; nakade3[0][1] = 1;              nakade3[0][2] = 2;
    nakade3[1][0] = 0; nakade3[1][1] = board_size;     nakade3[1][2] = 2 * board_size;
    nakade3[2][0] = 0; nakade3[2][1] = 1;              nakade3[2][2] = board_size + 1;
    nakade3[3][0] = 0; nakade3[3][1] = board_size - 1; nakade3[3][2] = board_size;
    nakade3[4][0] = 0; nakade3[4][1] = board_size;     nakade3[4][2] = board_size + 1;
    nakade3[5][0] = 0; nakade3[5][1] = 1;              nakade3[5][2] = board_size;

    nakade4[0][0] = 0;              nakade4[0][1] = board_size - 1;
    nakade4[0][2] = board_size;     nakade4[0][3] = board_size + 1;
    nakade4[1][0] = 0;              nakade4[1][1] = board_size - 1;
    nakade4[1][2] = board_size;     nakade4[1][3] = 2 * board_size;
    nakade4[2][0] = 0;              nakade4[2][1] = board_size;
    nakade4[2][2] = board_size + 1; nakade4[2][3] = 2 * board_size;
    nakade4[3][0] = 0;              nakade4[3][1] = 1;
    nakade4[3][2] = 2;              nakade4[3][3] = board_size + 1;
    nakade4[4][0] = 0;              nakade4[4][1] = 1;
    nakade4[4][2] = board_size;     nakade4[4][3] = board_size + 1;

    nakade5[0][0] = 0;                  nakade5[0][1] = board_size - 1; nakade5[0][2] = board_size;
    nakade5[0][3] = board_size + 1;     nakade5[0][4] = 2 * board_size;
    nakade5[1][0] = 0;                  nakade5[1][1] = board_size - 1; nakade5[1][2] = board_size;
    nakade5[1][3] = 2 * board_size - 1; nakade5[1][4] = 2 * board_size;
    nakade5[2][0] = 0;                  nakade5[2][1] = 1;              nakade5[2][2] = board_size;
    nakade5[2][3] = board_size + 1;     nakade5[2][4] = board_size + 2;
    nakade5[3][0] = 0;                  nakade5[3][1] = 1;              nakade5[3][2] = board_size;
    nakade5[3][3] = board_size + 1;     nakade5[3][4] = 2 * board_size;
    nakade5[4][0] = 0;                  nakade5[4][1] = 1;              nakade5[4][2] = 2;
    nakade5[4][3] = board_size + 1;     nakade5[4][4] = board_size + 2;
    nakade5[5][0] = 0;                  nakade5[5][1] = board_size;     nakade5[5][2] = board_size + 1;
    nakade5[5][3] = 2 * board_size;     nakade5[5][4] = 2 * board_size + 1;
    nakade5[6][0] = 0;                  nakade5[6][1] = 1;              nakade5[6][2] = 2;
    nakade5[6][3] = board_size;         nakade5[6][4] = board_size + 1;
    nakade5[7][0] = 0;                  nakade5[7][1] = 1;              nakade5[7][2] = board_size;
    nakade5[7][3] = board_size + 1;     nakade5[7][4] = 2 * board_size + 1;
    nakade5[8][0] = 0;                  nakade5[8][1] = 1;              nakade5[8][2] = board_size - 1;
    nakade5[8][3] = board_size;         nakade5[8][4] = board_size + 1;

    nakade6[0][0] = 0;                  nakade6[0][1] = board_size - 1;
    nakade6[0][2] = board_size;         nakade6[0][3] = board_size + 1;
    nakade6[0][4] = 2 * board_size - 1; nakade6[0][5] = 2 * board_size;
    nakade6[1][0] = 0;                  nakade6[1][1] = 1;
    nakade6[1][2] = board_size;         nakade6[1][3] = board_size + 1;
    nakade6[1][4] = board_size + 2;     nakade6[1][5] = 2 * board_size;
    nakade6[2][0] = 0;                  nakade6[2][1] = 1;
    nakade6[2][2] = board_size - 1;     nakade6[2][3] = board_size;
    nakade6[2][4] = board_size + 1;     nakade6[2][5] = 2 * board_size;
    nakade6[3][0] = 0;                  nakade6[3][1] = board_size - 1;
    nakade6[3][2] = board_size;         nakade6[3][3] = board_size + 1;
    nakade6[3][4] = 2 * board_size;     nakade6[3][5] = 2 * board_size + 1;

    nakade_pos[0][0] = 1;
    nakade_pos[0][1] = board_size;
    nakade_pos[0][2] = 1;
    nakade_pos[0][3] = board_size;
    nakade_pos[0][4] = board_size;
    nakade_pos[0][5] = 0;

    nakade_pos[1][0] = board_size;
    nakade_pos[1][1] = board_size;
    nakade_pos[1][2] = board_size;
    nakade_pos[1][3] = 1;
    nakade_pos[1][4] = 0;

    nakade_pos[2][0] = board_size;
    nakade_pos[2][1] = board_size;
    nakade_pos[2][2] = board_size + 1;
    nakade_pos[2][3] = board_size;
    nakade_pos[2][4] = 1;
    nakade_pos[2][5] = board_size;
    nakade_pos[2][6] = 1;
    nakade_pos[2][7] = board_size + 1;
    nakade_pos[2][8] = board_size;

    nakade_pos[3][0] = board_size;
    nakade_pos[3][1] = board_size + 1;
    nakade_pos[3][2] = board_size;
    nakade_pos[3][3] = board_size;

    for i in xrange(NAKADE_3):
        nakade_hash[0][i] = 0
        for j in range(3):
            nakade_hash[0][i] ^= shape_bit[start + nakade3[i][j]]

    for i in xrange(NAKADE_4):
        nakade_hash[1][i] = 0
        for j in xrange(4):
            nakade_hash[1][i] ^= shape_bit[start + nakade4[i][j]]

    for i in xrange(NAKADE_5):
        nakade_hash[2][i] = 0
        for j in xrange(5):
            nakade_hash[2][i] ^= shape_bit[start + nakade5[i][j]]

    for i in xrange(NAKADE_6):
        nakade_hash[3][i] = 0
        for j in xrange(6):
            nakade_hash[3][i] ^= shape_bit[start + nakade6[i][j]]


def sgftest(search_name=None):
    initialize_hash()
    import glob
    from AlphaGo.util import sgf_iter_states
    if not search_name:
        search_name = '../dataset/kihuu/*.sgf'
    for file_name in glob.glob(search_name):
        try:
            with open(file_name, 'r')  as file_object:
                state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

            for i, (state, move, player) in enumerate(state_action_iterator):
                nakade_pos = search_nakade(state)
                if nakade_pos:
                    print file_name
                    return
        except:
            pass
