cimport go

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_


def test_set_board_size_9():
    go.set_board_size(9)

    ok_(go.pure_board_size == 9)
    ok_(go.pure_board_max == 81)
    ok_(go.board_size == 13)
    ok_(go.board_max == 169)
    ok_(go.max_string == 64)
    ok_(go.max_neighbor == 64)
    ok_(go.board_start == 2)
    ok_(go.board_end == 10)
    ok_(go.string_lib_max == 143)
    ok_(go.string_pos_max == 143)
    ok_(go.string_end == 142)
    ok_(go.neighbor_end == 63)
    ok_(go.liberty_end == 142)
    ok_(go.max_records == 243)
    ok_(go.max_moves == 242)


def test_set_board_size_19():
    go.set_board_size(19)

    ok_(go.pure_board_size == 19)
    ok_(go.pure_board_max == 361)
    ok_(go.board_size == 23)
    ok_(go.board_max == 529)
    ok_(go.max_string == 288)
    ok_(go.max_neighbor == 288)
    ok_(go.board_start == 2)
    ok_(go.board_end == 20)
    ok_(go.string_lib_max == 483)
    ok_(go.string_pos_max == 483)
    ok_(go.string_end == 482)
    ok_(go.neighbor_end == 287)
    ok_(go.liberty_end == 482)
    ok_(go.max_records == 1083)
    ok_(go.max_moves == 1082)


def test_add_liberty_to_isolated_one():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int neighbor4[4]

    game = __initialize_game()

    origin = go.POS(4, 4, go.board_size)
    string = __initialize_string(game, origin, go.S_BLACK) 

    go.get_neighbor4(neighbor4, origin)

    go.add_liberty(string, neighbor4[0], 0)
    ok_(string.lib[0] == neighbor4[0])
    ok_(string.libs == 1)

    go.add_liberty(string, neighbor4[1], neighbor4[0])
    ok_(string.lib[neighbor4[0]] == neighbor4[1])
    ok_(string.libs == 2)

    go.add_liberty(string, neighbor4[2], neighbor4[1])
    ok_(string.lib[neighbor4[1]] == neighbor4[2])
    ok_(string.libs == 3)

    go.add_liberty(string, neighbor4[3], neighbor4[2])
    ok_(string.lib[neighbor4[2]] == neighbor4[3])
    ok_(string.libs == 4)


def test_remove_liberty_of_isolated_one():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int neighbor4[4]

    game = __initialize_game()

    pos = go.POS(4, 4, go.board_size)
    go.make_string(game, pos, go.S_BLACK)
    string = &game.string[1]

    go.get_neighbor4(neighbor4, pos)
    north = neighbor4[0]
    west  = neighbor4[1]
    east  = neighbor4[2]
    south = neighbor4[3]

    go.remove_liberty(string, north)
    ok_(string.lib[north] == 0)
    ok_(string.lib[0] == west)

    go.remove_liberty(string, west)
    ok_(string.lib[west] == 0)
    ok_(string.lib[0] == east)

    go.remove_liberty(string, east)
    ok_(string.lib[east] == 0)
    ok_(string.lib[0] == south)

    go.remove_liberty(string, south)
    ok_(string.lib[south] == 0)


def test_make_string_of_isolated_one():
    cdef go.game_state_t *game
    cdef go.string_t *new_string

    game = __initialize_game()

    pos = go.POS(4, 4, go.board_size)
    go.make_string(game, pos, go.S_BLACK)

    new_string = &game.string[1]

    ok_(new_string.lib[0] == pos - go.board_size)
    ok_(new_string.lib[pos - go.board_size] == pos - 1)
    ok_(new_string.lib[pos - 1] == pos + 1)
    ok_(new_string.lib[pos + 1] == pos + go.board_size)
    ok_(new_string.lib[pos + go.board_size] == go.string_lib_max - 1)
    ok_(new_string.libs == 4)

    ok_(game.string_id[pos] == 1)
    ok_(game.string_next[pos] == go.string_pos_max - 1)


def test_add_neighbor_to_isolated_one():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int neighbor[4] 

    game = __initialize_game()

    pos = go.POS(4, 4, go.board_size)
    go.make_string(game, pos, go.S_BLACK)

    go.get_neighbor4(neighbor, pos)
    south = neighbor[0]

    string = __initialize_string(game, south, go.S_WHITE)

    string_id = game.string_id[pos]
    neighbor_string_id = game.string_id[south]

    ok_(string_id == 1)
    ok_(neighbor_string_id == 2)
    ok_(game.string[1].neighbor[0] == go.max_neighbor - 1)
    ok_(game.string[2].neighbor[0] == go.max_neighbor - 1)

    go.add_neighbor(&game.string[string_id], neighbor_string_id, 0)
    go.add_neighbor(&game.string[neighbor_string_id], string_id, 0)

    ok_(game.string[1].neighbor[0] == 2)
    ok_(game.string[1].neighbor[2] == go.max_neighbor - 1)
    ok_(game.string[2].neighbor[0] == 1)
    ok_(game.string[2].neighbor[1] == go.max_neighbor - 1)


def test_add_stone_to_string_less_position():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int origin = go.POS(4, 4, go.board_size), add_stone = origin - 1, head = 0

    game = __initialize_game()

    go.make_string(game, origin, go.S_BLACK)

    ok_(game.string_id[origin] == 1)

    string = &game.string[1]
    go.add_stone_to_string(game, string, add_stone, head)

    ok_(string.size == 2)
    ok_(string.origin == add_stone)
    ok_(game.string_next[add_stone] == origin)
    ok_(game.string_next[origin] == go.string_pos_max - 1)


def test_add_stone_to_string_larger_position_from_zero_head():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int origin = go.POS(4, 4, go.board_size), add_stone = origin + 1, head = 0

    game = __initialize_game()

    go.make_string(game, origin, go.S_BLACK)

    ok_(game.string_id[origin] == 1)

    string = &game.string[1]
    go.add_stone_to_string(game, string, add_stone, head)

    ok_(string.size == 2)
    ok_(game.string_next[origin] == add_stone)
    ok_(game.string_next[add_stone] == go.string_pos_max - 1)


def test_add_stone_to_string_larger_position_from_nonzero_head():
    pass


def test_add_stone_to_isolated_stone():
    cdef go.game_state_t *game
    cdef go.string_t *string
    cdef int origin = go.POS(4, 4, go.board_size)
    cdef int add_pos = origin + 1

    game = __initialize_game()

    go.make_string(game, origin, go.S_BLACK)

    ok_(game.string_id[origin] == 1)
    
    string = &game.string[1]
    ok_(string.libs == 4)
    ok_(string.size == 1)

    go.add_stone(game, add_pos, go.S_BLACK, 1)

    ok_(string.libs == 8)
    ok_(string.size == 2)
    ok_(game.string_id[add_pos] == 1)
    ok_(game.string_next[origin] == add_pos)
    ok_(game.string_next[add_pos] == go.string_pos_max - 1)


def test_merge_string_two():
    cdef go.game_state_t *game
    cdef int add_pos = go.POS(4, 4, go.board_size)
    cdef int first = add_pos - 1
    cdef int second = add_pos + 1
    cdef go.string_t *string[3]

    game = __initialize_game()

    go.make_string(game, first, go.S_BLACK)
    go.make_string(game, second, go.S_BLACK)

    ok_(game.string_id[first], 1)
    ok_(game.string_id[second], 2)

    ok_(game.string_next[first], go.string_pos_max - 1)
    ok_(game.string_next[second], go.string_pos_max - 1)

    ok_(game.string[1].neighbor[0] == go.max_neighbor - 1)
    ok_(game.string[2].neighbor[0] == go.max_neighbor - 1)

    ok_(game.string[1].lib[first + go.board_size] == go.string_lib_max - 1)
    ok_(game.string[2].lib[second + go.board_size] == go.string_lib_max - 1)

    string[0] = &game.string[2]

    go.add_stone(game, add_pos, go.S_BLACK, 1)
    
    ok_(game.string_next[first], add_pos)
    ok_(game.string_next[add_pos], go.string_pos_max - 1)
    ok_(game.string_next[second], go.string_pos_max - 1)

    go.merge_string(game, &game.string[1], string, 1)


def test_merge_string_three():
    cdef go.game_state_t *game
    cdef int add_pos = go.POS(4, 4, go.board_size)
    cdef int first = add_pos - go.board_size
    cdef int second = add_pos - 1
    cdef int third = add_pos + 1
    cdef go.string_t *string[3]

    game = __initialize_game()

    go.make_string(game, first, go.S_BLACK)
    go.make_string(game, second, go.S_BLACK)
    go.make_string(game, third, go.S_BLACK)

    ok_(game.string_id[first], 1)
    ok_(game.string_id[second], 2)
    ok_(game.string_id[third], 3)

    ok_(game.string_next[first], go.string_pos_max - 1)
    ok_(game.string_next[second], go.string_pos_max - 1)
    ok_(game.string_next[third], go.string_pos_max - 1)

    ok_(game.string[1].neighbor[0] == go.max_neighbor - 1)
    ok_(game.string[2].neighbor[0] == go.max_neighbor - 1)
    ok_(game.string[3].neighbor[0] == go.max_neighbor - 1)

    ok_(game.string[1].lib[first + go.board_size] == go.string_lib_max - 1)
    ok_(game.string[2].lib[second + go.board_size] == go.string_lib_max - 1)
    ok_(game.string[3].lib[third + go.board_size] == go.string_lib_max - 1)

    string[0] = &game.string[2]
    string[1] = &game.string[3]

    go.add_stone(game, add_pos, go.S_BLACK, 1)
    
    ok_(game.string_next[first], add_pos)
    ok_(game.string_next[add_pos], go.string_pos_max - 1)
    ok_(game.string_next[second], go.string_pos_max - 1)
    ok_(game.string_next[third], go.string_pos_max - 1)

    go.merge_string(game, &game.string[1], string, 1)


cdef go.game_state_t* __initialize_game(int board_size=9):
    cdef go.game_state_t *game

    go.set_board_size(board_size)

    game = go.allocate_game()
    go.initialize_board(game)

    return game


cdef go.string_t* __initialize_string(go.game_state_t *game, int origin, char color):
    cdef go.string_t *string
    cdef string_id = 1

    while game.string[string_id].flag:
        string_id += 1

    string = &game.string[string_id]

    go.fill_n_short(string.lib, go.string_lib_max, 0)
    go.fill_n_short(string.neighbor, go.max_neighbor, 0)
    string.lib[0] = go.string_lib_max - 1
    string.neighbor[0] = go.max_neighbor - 1
    string.libs = 0
    string.color = color
    string.origin = origin 
    string.size = 1
    string.neighbors = 0

    game.string_id[origin] = string_id
    game.string_next[origin] = go.string_pos_max - 1

    return string

