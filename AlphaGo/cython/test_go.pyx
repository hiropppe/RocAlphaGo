cimport go

from nose.tools import ok_, eq_


def test_set_board_size_9():
    go.set_board_size(9)

    ok_(go.pure_board_size == 9)
    ok_(go.pure_board_max == 81)
    ok_(go.board_size == 13)
    ok_(go.board_max == 169)
    ok_(go.max_string == 7)
    ok_(go.max_neighbor == 7)
    ok_(go.board_start == 2)
    ok_(go.board_end == 10)
    ok_(go.string_lib_max == 143)
    ok_(go.string_pos_max == 143)
    ok_(go.string_end == 142)
    ok_(go.neighbor_end == 6)
    ok_(go.liberty_end == 142)
    ok_(go.max_records == 27)
    ok_(go.max_moves == 26)


def test_set_board_size_19():
    go.set_board_size(19)

    ok_(go.pure_board_size == 19)
    ok_(go.pure_board_max == 361)
    ok_(go.board_size == 23)
    ok_(go.board_max == 529)
    ok_(go.max_string == 15)
    ok_(go.max_neighbor == 15)
    ok_(go.board_start == 2)
    ok_(go.board_end == 20)
    ok_(go.string_lib_max == 483)
    ok_(go.string_pos_max == 483)
    ok_(go.string_end == 482)
    ok_(go.neighbor_end == 14)
    ok_(go.liberty_end == 482)
    ok_(go.max_records == 57)
    ok_(go.max_moves == 56)


def test_make_string():
    cdef go.string_t *new_string

    go.set_board_size(9)

    game = go.allocate_game()
    go.initialize_board(game)

    pos = go.POS(4, 4, go.board_size)
    go.make_string(game, 56, 1)

    new_string = &game.string[1]

    ok_(new_string.lib[0] == go.liberty_end)

