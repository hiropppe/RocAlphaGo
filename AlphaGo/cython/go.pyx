# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

# cimport point

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport exp as cexp

from libcpp.string cimport string as cppstring

# global variables
gogui_x[:] = ['I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
              'U', 'V', 'W', 'X', 'Y', 'Z']

pure_board_size = PURE_BOARD_SIZE
pure_board_max = PURE_BOARD_MAX

board_size = BOARD_SIZE
board_max = BOARD_MAX

max_string = MAX_STRING
max_neighbor = MAX_STRING

board_start = BOARD_START
board_end = BOARD_END

string_lib_max = STRING_LIB_MAX
string_pos_max = STRING_POS_MAX

string_end = STRING_END
neighbor_end = NEIGHBOR_END
liberty_end = LIBERTY_END

max_records = MAX_RECORDS
max_moves = MAX_MOVES

default_komi = KOMI


cdef void fill_n_char (char *arr, int size, char v):
    cdef int i
    for i in range(size):
        arr[i] = v


cdef void fill_n_short (short *arr, int size, short v):
    cdef int i
    for i in range(size):
        arr[i] = v


cdef void fill_n_int (int *arr, int size, int v):
    cdef int i
    for i in range(size):
        arr[i] = v


cdef game_state_t *allocate_game():
    cdef game_state_t *game

    game = <game_state_t *>malloc(sizeof(game_state_t))
    memset(game, 0, sizeof(game_state_t))

    return game


cdef void initialize_board(game_state_t *game):
    cdef int i, x, y, pos

    game.moves = 1
    game.ko_pos = 0
    game.ko_move = 0
    game.pass_count = 0
    game.current_hash = 0

    fill_n_char(game.board, BOARD_MAX, 0)
    fill_n_int(game.candidates, BOARD_MAX, 0)
    # fill_n_int(game.string_id, STRING_POS_MAX, 0)
    # fill_n_int(game.string_next, STRING_POS_MAX, 0)
    # fill_n_int(game.prisoner, S_MAX, 0)
    

    for y in range(board_size):
        for x in range(OB_SIZE):
            game.board[POS(x, y, board_size)] = S_OB
            game.board[POS(y, x, board_size)] = S_OB
            game.board[POS(y, board_size - 1 - x, board_size)] = S_OB
            game.board[POS(board_size - 1 - x, y, board_size)] = S_OB

    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_end + 1):
            pos = POS(x, y, board_size)
            game.candidates[pos] = 1 

    for i in range(max_string):
        game.string[i].flag = False


cdef void do_move(game_state_t *game, int pos, char color):
    cdef char other = FLIP_COLOR(color)
    cdef int connection = 0
    cdef int connect[4]
    cdef int prisoner = 0
    cdef int neighbor4[4]
    cdef int neighbor_pos, neighbor_string_id
    cdef string_t *neighbor_string
    cdef int i

    connect[:] = [0, 0, 0, 0]

    game.capture_num[color] = 0

    if game.moves < max_records:
        game.record[game.moves].color = color
        game.record[game.moves].pos = pos

    if pos == PASS:
        game.moves += 1
        return

    game.board[pos] = color

    game.candidates[pos] = 0

    get_neighbor4(neighbor4, pos)

    for i in range(4):
        neighbor_pos = neighbor4[i]
        neighbor_string_id = game.string_id[neighbor_pos]
        neighbor_string = &game.string[neighbor_string_id]
        if game.board[neighbor_pos] == color:
            remove_liberty(neighbor_string, pos)
            connect[connection] = neighbor_string_id
            connection += 1
        elif game.board[neighbor_pos] == other:
            remove_liberty(neighbor_string, pos)
            if game.string[game.string_id[neighbor_pos]].libs == 0:
                prisoner += remove_string(game, neighbor_string)

    game.prisoner[color] += prisoner

    if connection == 0:
        make_string(game, pos, color)
        if prisoner == 1 and game.string[game.string_id[pos]].libs == 1:
            pass
    elif connection == 1:
        add_stone(game, pos, color, connect[0])
    else:
        connect_string(game, pos, color, connection, connect)

    game.moves += 1


cdef void connect_string(game_state_t *game, int pos, char color, int connection, int string_id[4]):
    cdef min_string_id = string_id[0]
    cdef string_t *string[3]
    cdef int connections = 0
    cdef bint flag = True
    cdef int i, j

    # print 'ConnectString', pos, color, connection

    for i in range(1, connection):
        flag = True
        for j in range(i):
            if string_id[j] == string_id[i]:
                flag = False
                break
        if flag:
            if min_string_id > string_id[i]:
                string[connections] = &game.string[min_string_id]
                min_string_id = string_id[i]
            else:
                string[connections] = &game.string[string_id[i]]
            connections += 1

    add_stone(game, pos, color, min_string_id)

    if connections > 0:
        merge_string(game, &game.string[min_string_id], string, connections)


cdef void merge_string(game_state_t *game, string_t *dst, string_t *src[3], int n):
    cdef int tmp, pos, prev, neighbor
    cdef int string_id = game.string_id[dst.origin]
    cdef int removed_string_id
    cdef int i

    # print 'MergeString', n

    for i in range(n):
        removed_string_id = game.string_id[src[i].origin]

        prev = 0
        pos = src[i].lib[0]
        while pos != liberty_end:
            prev = add_liberty(dst, pos, prev)
            pos = src[i].lib[pos]

        prev = 0
        pos = src[i].origin
        while pos != string_end:
            game.string_id[pos] = string_id
            tmp = game.string_next[pos]
            add_stone_to_string(game, dst, pos, prev)
            prev = pos
            pos = tmp

        prev = 0
        neighbor = src[i].neighbor[0]
        while neighbor != neighbor_end:
            remove_neighbor_string(&game.string[neighbor], removed_string_id)
            add_neighbor(dst, neighbor, prev)
            add_neighbor(&game.string[neighbor], string_id, prev)
            prev = neighbor
            neighbor = src[i].neighbor[neighbor]

        src[i].flag = False


cdef void add_stone(game_state_t *game, int pos, char color, int string_id):
    cdef string_t *add_string
    cdef int lib_add = 0
    cdef int other = FLIP_COLOR(color)
    cdef int neighbor4[4]
    cdef int neighbor_pos, neighbor_string_id
    cdef int i

    # print 'AddStone', pos, color, string_id

    game.string_id[pos] = string_id

    add_string = &game.string[string_id]

    add_stone_to_string(game, add_string, pos, 0)

    get_neighbor4(neighbor4, pos)

    for i in range(4):
        neighbor_pos = neighbor4[i]
        if game.board[neighbor_pos] == S_EMPTY:
            lib_add = add_liberty(add_string, neighbor_pos, lib_add)
        elif game.board[neighbor_pos] == other:
            neighbor_string_id = game.string_id[neighbor_pos]
            add_neighbor(&game.string[neighbor_string_id], string_id, 0)
            add_neighbor(&game.string[string_id], neighbor_string_id, 0)


cdef void add_stone_to_string(game_state_t *game, string_t *string, int pos, int head):
    cdef int string_pos

    # print 'AddStoneToString', pos, string.origin, head

    if pos == string_end:
        return

    if string.origin > pos:
        game.string_next[pos] = string.origin
        string.origin = pos
    else:
        if head != 0:
            string_pos = head
        else:
            string_pos = string.origin

        while game.string_next[string_pos] < pos:
            string_pos = game.string_next[string_pos]

        game.string_next[pos] = game.string_next[string_pos]
        game.string_next[string_pos] = pos

    # print 'Assert String', string.origin, game.string_next[pos], game.string_next[string_pos]
    # string_pos = string.origin
    # while game.string_next[string_pos] != string_end:
    #    string_pos = game.string_next[string_pos]

    string.size += 1


cdef void make_string(game_state_t *game, int pos, char color):
    cdef string_t *new_string
    cdef int string_id = 1
    cdef int lib_add = 0
    cdef int other = FLIP_COLOR(color)
    cdef int neighbor4[4]
    cdef int neighbor_pos
    cdef int neighbor_string_id
    cdef int i

    while game.string[string_id].flag:
        string_id += 1

    # print 'MakeString', string_id, pos, color

    new_string = &game.string[string_id]

    fill_n_short(new_string.lib, string_lib_max, 0)
    fill_n_short(new_string.neighbor, max_neighbor, 0)

    new_string.lib[0] = liberty_end
    new_string.neighbor[0] = neighbor_end
    new_string.libs = 0
    new_string.color = color
    new_string.origin = pos
    new_string.size = 1
    new_string.neighbors = 0

    game.string_id[pos] = string_id
    game.string_next[pos] = string_end

    get_neighbor4(neighbor4, pos)

    for i in range(4):
        neighbor_pos = neighbor4[i]
        if game.board[neighbor_pos] == S_EMPTY:
            lib_add = add_liberty(new_string, neighbor_pos, lib_add)
        elif game.board[neighbor_pos] == other:
            neighbor_string_id = game.string_id[neighbor_pos]
            add_neighbor(&game.string[neighbor_string_id], string_id, 0)
            add_neighbor(&game.string[string_id], neighbor_string_id, 0)

    new_string.flag = True


cdef int remove_string(game_state_t *game, string_t *string):
    cdef int next
    cdef int neighbor
    cdef int pos = string.origin
    cdef int remove_string_id = game.string_id[pos]
    cdef int remove_color = game.board[pos]

    # print 'RemoveString', remove_string_id

    while True:
        game.board[pos] = S_EMPTY

        north_string_id = game.string_id[NORTH(pos, board_size)]
        west_string_id = game.string_id[WEST(pos)]
        east_string_id = game.string_id[EAST(pos)]
        south_string_id = game.string_id[SOUTH(pos, board_size)]

        if game.string[north_string_id].flag:
            add_liberty(&game.string[north_string_id], pos, 0)
        if game.string[west_string_id].flag:
            add_liberty(&game.string[west_string_id], pos, 0)
        if game.string[east_string_id].flag:
            add_liberty(&game.string[east_string_id], pos, 0)
        if game.string[south_string_id].flag:
            add_liberty(&game.string[south_string_id], pos, 0)

        next = game.string_next[pos]

        game.string_next[pos] = 0
        game.string_id[pos] = 0

        pos = next

        if pos == string_end:
            break

    neighbor = string.neighbor[0]
    while neighbor != neighbor_end:
        remove_neighbor_string(&game.string[neighbor], remove_string_id)
        neighbor = string.neighbor[neighbor]

    string.flag = False

    return string.size


cdef int add_liberty(string_t *string, int pos, int head):
    cdef int lib

    # print 'AddLiberty', pos, head

    if string.lib[pos] != 0:
        return pos

    lib = head

    while string.lib[lib] < pos:
        lib = string.lib[lib]

    string.lib[pos] = string.lib[lib]
    string.lib[lib] = <short>pos

    string.libs += 1

    return pos


cdef void remove_liberty(string_t *string, int pos):
    cdef int lib = 0

    # print 'RemoveLiberty', pos, string.lib[pos]

    if string.lib[pos] == 0:
        return

    while string.lib[lib] != pos:
        lib = string.lib[lib]

    string.lib[lib] = string.lib[string.lib[lib]]
    string.lib[pos] = <short>0

    string.libs -= 1


cdef void add_neighbor(string_t *string, int string_id, int head):
    cdef int neighbor = 0

    # print 'AddNeighbor', string.neighbor[neighbor], string_id, head

    if string.neighbor[string_id] != 0:
        return

    neighbor = head

    while string.neighbor[neighbor] < string_id:
        neighbor = string.neighbor[neighbor]

    string.neighbor[string_id] = string.neighbor[neighbor]
    string.neighbor[neighbor] = <short>string_id

    string.neighbors += 1


cdef void remove_neighbor_string(string_t *string, int string_id):
    cdef int neighbor = 0

    # print 'RemoveNeighborString', string.neighbor[string_id], string_id

    if string.neighbor[string_id] == 0:
        return

    while string.neighbor[neighbor] != string_id:
        neighbor = string.neighbor[neighbor]

    string.neighbor[neighbor] = string.neighbor[string.neighbor[neighbor]]
    string.neighbor[string_id] = 0

    string.neighbors -= 1


cdef void get_neighbor4(int neighbor4[4], int pos):
    neighbor4[0] = NORTH(pos, board_size)
    neighbor4[1] = WEST(pos)
    neighbor4[2] = EAST(pos)
    neighbor4[3] = SOUTH(pos, board_size)


cdef void init_board_position():
    cdef int i, x, y, p

    global onboard_pos, board_x, board_y

    free(onboard_pos)
    free(board_x)
    free(board_y)

    onboard_pos = <int *>malloc(pure_board_max * sizeof(int))
    board_x = <int *>malloc(board_max * sizeof(int))
    board_y = <int *>malloc(board_max * sizeof(int))
    i = 0
    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_end + 1):
            p = POS(x, y, board_size)
            onboard_pos[i] = p
            board_x[p] = x
            board_y[p] = y
            i += 1


cdef void init_line_number():
    cdef int x, y, p, d

    global board_dis_x, board_dis_y

    free(board_dis_x)
    free(board_dis_y)

    board_dis_x = <int *>malloc(board_max * sizeof(int))
    board_dis_y = <int *>malloc(board_max * sizeof(int))
    for y in range(board_start, board_end + 1):
        for x in range(board_start, board_start + pure_board_size / 2 + 1):
            d = x - (OB_SIZE - 1)
            board_dis_x[POS(x, y, board_size)] = d
            board_dis_x[POS(board_end + OB_SIZE - x, y, board_size)] = d
            board_dis_y[POS(y, x, board_size)] = d
            board_dis_y[POS(y, board_end + OB_SIZE - x, board_size)] = d 


cdef void init_move_distance():
    cdef int x, y

    global move_dis

    move_dis = np.zeros((pure_board_size, pure_board_size), dtype=np.int32)
    for y in range(pure_board_size):
        for x in range(pure_board_size):
            move_dis[x, y] = x + y + MAX(x, y)
            if move_dis[x, y] > MOVE_DISTANCE_MAX:
                move_dis[x, y] = MOVE_DISTANCE_MAX


cdef void init_board_position_id():
    cdef int i, x, y

    global board_pos_id

    free(board_pos_id)

    board_pos_id = <int *>malloc(board_max * sizeof(int))
    fill_n_int(board_pos_id, board_max, 0)
    i = 1
    for y in range(board_start, board_start + pure_board_size / 2 + 1):
        for x in range(board_start, y + 1):
            board_pos_id[POS(x, y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - x, y, board_size)] = i
            board_pos_id[POS(y, x, board_size)] = i
            board_pos_id[POS(y, board_end + OB_SIZE - x, board_size)] = i
            board_pos_id[POS(x, board_end + OB_SIZE - y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - x, board_end + OB_SIZE - y, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - y, x, board_size)] = i
            board_pos_id[POS(board_end + OB_SIZE - y, board_end + OB_SIZE - x, board_size)] = i
            i+=1


cdef void init_corner():
    global corner, corner_neighbor

    free(corner)

    corner = <int *>malloc(4 * sizeof(int))
    corner[0] = POS(board_start, board_start, board_size)
    corner[1] = POS(board_start, board_end, board_size)
    corner[2] = POS(board_end, board_start, board_size)
    corner[3] = POS(board_end, board_end, board_size)

    corner_neighbor = np.zeros((4, 2), dtype=np.int32)
    corner_neighbor[0, 0] = EAST(corner[0])
    corner_neighbor[0, 1] = SOUTH(corner[0], board_size)
    corner_neighbor[1, 0] = NORTH(corner[1], board_size)
    corner_neighbor[1, 1] = EAST(corner[1])
    corner_neighbor[2, 0] = WEST(corner[2])
    corner_neighbor[2, 1] = SOUTH(corner[2], board_size)
    corner_neighbor[3, 0] = NORTH(corner[3], board_size)
    corner_neighbor[3, 1] = WEST(corner[3])


cdef void initialize_neighbor():
    pass


cdef void initialize_const():
    global komi

    komi = <double *>malloc(S_OB * sizeof(double))
    komi[0] = default_komi
    komi[<int>S_BLACK] = default_komi + 1.0
    komi[<int>S_WHITE] = default_komi - 1.0

    init_board_position()

    init_line_number()

    init_move_distance()

    init_board_position_id()

    init_corner()


cdef void set_board_size(int size):
    global pure_board_size, pure_board_max
    global board_size, board_max
    global max_string, max_neighbor
    global board_start, board_end
    global string_lib_max, string_pos_max
    global string_end, neighbor_end, liberty_end
    global max_records, max_moves

    pure_board_size = size
    pure_board_max = size ** 2

    board_size = size + 2 * OB_SIZE
    board_max = board_size ** 2

    max_string = pure_board_max * 4 / 5
    max_neighbor = max_string

    board_start = OB_SIZE
    board_end = pure_board_size + OB_SIZE - 1

    string_lib_max = board_size * (size + OB_SIZE)
    string_pos_max = board_size * (size + OB_SIZE)

    string_end = string_pos_max - 1
    neighbor_end = max_neighbor - 1
    liberty_end = string_lib_max - 1

    max_records = pure_board_max * 3
    max_moves = max_records - 1

    initialize_const()



## Test bench
cpdef void arr_bench():
    arr_sum()
    arrmv_sum()
    numpy_sum()
    numpymv_sum()


cpdef void arr_sum():
    cdef int i, tmp, sum = 0
    cdef int *arr
    import time
    
    arr = <int *>malloc(100000 * sizeof(int))
    fill_n_int(arr, 100000, 0)

    s = time.time()
    for i in range(100000):
        arr[i] = 1
    print 'Array write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = arr[i-1]
    print 'Array read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += arr[i]
    print 'Array sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void arrmv_sum():
    cdef int i, tmp, sum = 0
    cdef int *arr
    cdef int[:] mv
    import time
    
    arr = <int *>malloc(100000 * sizeof(int))
    mv = <int[:100000]>arr
    mv[...] = 0

    s = time.time()
    for i in range(100000):
        mv[i] = 1
    print 'Memoryview of c array write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = mv[i-1]
    print 'Memoryview of c array read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += mv[i]
    print 'Memoryview of c array sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void numpy_sum():
    cdef int i, tmp, sum = 0
    import time

    arr = np.ndarray((100000), dtype=np.int32)
    arr.fill(0)

    s = time.time()
    for i in range(100000):
        arr[i] = 1 
    print 'ndarray write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = arr[i-1]
    print 'ndarray read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    sum = arr.sum()
    print 'ndarray sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void numpymv_sum():
    cdef int i, tmp, sum = 0
    cdef int[:] mv
    import time

    arr = np.ndarray((100000), dtype=np.int32)
    arr.fill(0)
    mv = arr

    s = time.time()
    for i in range(100000):
        mv[i] = 1
    print 'Memoryview of ndarray write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = mv[i-1]
    print 'Memoryview of ndarray read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += mv[i]
    print 'Memoryview of ndarray sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef test_playout(int n_playout=1, int move_limit=500):
    """ benchmark playout speed
    """
    cdef int i, n
    cdef char color = S_BLACK
    cdef list empties = []
    cdef list po_speeds = [], move_speeds = []
    cdef game_state_t *game

    import itertools
    import random
    import time

    set_board_size(19)
    for n in range(n_playout):
        game = allocate_game()
        initialize_board(game)
        pos_generator = itertools.product(range(board_start, board_end + 1), repeat=2)
        empties = [POS(p[0], p[1], board_size) for p in pos_generator]
        ps = time.time()
        for i in range(move_limit):
            if not empties:
                break
            pos = random.choice(empties)
            empties.remove(pos)
            ms = time.time()
            do_move(game, pos, color)
            color = FLIP_COLOR(color)
            move_speeds.append(time.time() - ms)
        po_speeds.append(time.time() - ps)

    print("Avg PO. {:.3f} us. ".format(np.mean(po_speeds)*1000*1000) +
          "Avg Move. {:.3f} ns. ".format(np.mean(move_speeds)*1000*1000*1000))

    print_board(game)


cdef void print_board(game_state_t *game):
    cdef char stone[<int>S_MAX]
    cdef int i, x, y, pos
    cdef list buf = []

    stone[:] = ['+', 'B', 'W', '#']

    buf.append("Prisoner(Black) : {:d}\n".format(game.prisoner[<int>S_BLACK]))
    buf.append("Prisoner(White) : {:d}\n".format(game.prisoner[<int>S_WHITE]))
    buf.append("Move : {:d}\n".format(game.moves))

    buf.append("    ")
    i = 1
    for _ in range(board_start, board_end + 1):
        buf.append(" {:s}".format(cppstring(1, <char>gogui_x[i])))
        i += 1
    buf.append("\n")

    buf.append("   +")
    for i in range(pure_board_size * 2 + 1):
        buf.append("-")
    buf.append("+\n")

    i = 1
    for y in range(board_start, board_end + 1):
        buf.append("{:d}:|".format(pure_board_size + 1 - i).rjust(4, ' '))
        for x in range(board_start, board_end + 1):
            pos = POS(x, y, board_size)
            # print pos, game.board[pos]
            buf.append(" {:s}".format(cppstring(1, stone[<int>game.board[pos]])))
        buf.append(" |\n")
        i += 1

    buf.append("   +")
    for i in range(1, pure_board_size * 2 + 1 + 1):
        buf.append("-")
    buf.append("+\n")

    print(''.join(buf))
