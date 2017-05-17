# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

from libcpp.string cimport string as cppstring

cimport go
cimport point


cdef void print_board(go.game_state_t *game):
    cdef char stone[<int>go.S_MAX]
    cdef int i, x, y, pos
    cdef list buf = []

    stone[:] = ['+', 'B', 'W', '#']

    buf.append("Prisoner(Black) : {:d}\n".format(game.prisoner[<int>go.S_BLACK]))
    buf.append("Prisoner(White) : {:d}\n".format(game.prisoner[<int>go.S_WHITE]))
    buf.append("Move : {:d}\n".format(game.moves))

    buf.append("    ")
    i = 1
    for _ in range(go.board_start, go.board_end + 1):
        buf.append(" {:s}".format(cppstring(1, <char>point.gogui_x[i])))
        i += 1
    buf.append("\n")

    buf.append("   +")
    for i in range(go.pure_board_size * 2 + 1):
        buf.append("-")
    buf.append("+\n")

    i = 1
    for y in range(go.board_start, go.board_end + 1):
        buf.append("{:d}:|".format(go.pure_board_size + 1 - i).rjust(4, ' '))
        for x in range(go.board_start, go.board_end + 1):
            pos = go.POS(x, y, go.board_size)
            buf.append(" {:s}".format(cppstring(1, stone[<int>game.board[pos]])))
        buf.append(" |\n")
        i += 1

    buf.append("   +")
    for i in range(1, go.pure_board_size * 2 + 1 + 1):
        buf.append("-")
    buf.append("+\n")

    print(''.join(buf))
