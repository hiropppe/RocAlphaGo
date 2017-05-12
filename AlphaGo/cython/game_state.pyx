# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport exp as cexp

cimport go


cdef class GameState:
    """
    cdef:
        char *board

        string_t *string
        int *string_id
        int *string_next

        readonly int moves
        move_t *record

        int *prisoner

        readonly int ko_pos
        readonly int ko_move

        readonly int pass_count

        readonly unsigned long long current_hash
    """

    def __cinit__(self):
        cdef int i, x, y

        self.board = <char *>malloc(go.BOARD_MAX * sizeof(char))
        self.string = <go.string_t *>malloc(go.MAX_STRING * sizeof(go.string_t))
        self.string_id = <int *>malloc(go.STRING_POS_MAX * sizeof(int))
        self.string_next = <int *>malloc(go.STRING_POS_MAX * sizeof(int))
        self.moves = 1
        self.record = <go.move_t *>malloc(go.MAX_RECORDS * sizeof(go.move_t))
        self.prisoner = <int *>malloc(go.S_MAX * sizeof(int))
        self.ko_pos = 0
        self.ko_move = 0
        self.pass_count = 0
        self.current_hash = 0

        go.fill_n_char(self.board, go.BOARD_MAX, 0)
        go.fill_n_int(self.string_id, go.STRING_POS_MAX, 0)
        go.fill_n_int(self.string_next, go.STRING_POS_MAX, 0)
        go.fill_n_int(self.prisoner, go.S_MAX, 0)

        for y in range(go.board_size):
            for x in range(go.OB_SIZE):
                self.board[go.POS(x, y, go.board_size)] = go.S_OB
                self.board[go.POS(y, x, go.board_size)] = go.S_OB
                self.board[go.POS(y, go.board_size - 1 - x, go.board_size)] = go.S_OB
                self.board[go.POS(go.board_size - 1 - x, y, go.board_size)] = go.S_OB

        for i in range(go.max_string):
            self.string[i].flag = False

    def __dealloc__(self):
        free(self.board)
        free(self.string)
        free(self.string_id)
        free(self.string_next)
        free(self.record)
        free(self.prisoner)

    cdef void do_move(self, int pos, char color):
        cdef char other = go.FLIP_COLOR(color)
        cdef int connection = 0
        cdef int connect[4]
        cdef int prisoner = 0
        cdef int neighbor4[4], neighbor_pos, neighbor_string_id
        cdef go.string_t *neighbor_string
        cdef int i

        connect[:] = [0, 0, 0, 0]
        if self.moves < go.max_records:
            self.record[self.moves].color = color
            self.record[self.moves].pos = pos

        if pos == go.PASS:
            self.moves += 1
            return

        self.board[pos] = color

        go.get_neighbor4(neighbor4, pos)

        for i in range(4):
            neighbor_pos = neighbor4[i]
            neighbor_string_id = self.string_id[neighbor_pos]
            neighbor_string = &self.string[neighbor_string_id]
            if self.board[neighbor_pos] == color:
                go.remove_liberty(neighbor_string, pos)
                connect[connection] = neighbor_string_id
                connection += 1
            elif self.board[neighbor_pos] == other:
                go.remove_liberty(neighbor_string, pos)
                if self.string[self.string_id[neighbor_pos]].libs == 0:
                    prisoner += self.remove_string(neighbor_string)

        if connection == 0:
            self.make_string(pos, color)
            if prisoner == 1 and self.string[self.string_id[pos]].libs == 1:
                pass
        elif connection == 1:
            self.add_stone(pos, color, connect[0])
        else:
            self.connect_string(pos, color, connection, connect)


    cdef void connect_string(self, int pos, char color, int connection, int string_id[4]):
        cdef min_string_id = string_id[0]
        cdef go.string_t *string[3]
        cdef int connections = 0
        cdef bint flag = True
        cdef int i, j

        print 'ConnectString', pos, color, connection

        for i in range(1, connection):
            flag = True
            for j in range(i):
                if string_id[j] == string_id[i]:
                    flag = False
                    break
            if flag:
                if min_string_id > string_id[i]:
                    string[connections] = &self.string[min]
                    min_string_id = string_id[i]
                else:
                    string[connections] = &self.string[string_id[i]]
                connections += 1

        self.add_stone(pos, color, min_string_id)

        if connections > 0:
            self.merge_string(&self.string[min_string_id], string, connections)

    cdef void merge_string(self, go.string_t *dst, go.string_t *src[3], int n):
        cdef int tmp, pos, prev, neighbor
        cdef int string_id = self.string_id[dst.origin]
        cdef int removed_string_id
        cdef int i

        for i in range(n):
            removed_string_id = self.string_id[src[i].origin]

            prev = 0
            pos = src[i].lib[0]
            while pos != go.LIBERTY_END:
                prev = go.add_liberty(dst, pos, prev)
                pos = src[i].lib[pos]

            prev = 0
            pos = src[i].origin
            while pos != go.STRING_END:
                self.string_id[pos] = string_id
                tmp = self.string_next[pos]
                self.add_stone_to_string(dst, pos, prev)
                prev = pos
                pos = tmp

            prev = 0
            neighbor = src[i].neighbor[0]
            while neighbor != go.NEIGHBOR_END:
                go.remove_neighbor_string(&self.string[neighbor], removed_string_id)
                go.add_neighbor(dst, neighbor, prev)
                go.add_neighbor(&self.string[neighbor], string_id, prev)
                prev = neighbor
                neighbor = src[i].neighbor[neighbor]

            src[i].flag = False

    cdef void add_stone(self, int pos, char color, int string_id):
        cdef go.string_t *add_string
        cdef int lib_add
        cdef int other = go.FLIP_COLOR(color)
        cdef int neighbor4[4], neighbor_pos, neighbor_string_id
        cdef int i

        print 'AddStone', pos, color, string_id

        self.string_id[pos] = string_id

        add_string = &self.string[string_id]

        self.add_stone_to_string(add_string, pos, 0)

        go.get_neighbor4(neighbor4, pos)

        for i in range(4):
            neighbor_pos = neighbor4[i]
            if self.board[neighbor_pos] == go.S_EMPTY:
                lib_add = go.add_liberty(add_string, neighbor_pos, lib_add)
            elif self.board[neighbor_pos] == other:
                neighbor_string_id = self.string_id[neighbor_pos]
                go.add_neighbor(&self.string[neighbor_string_id], string_id, 0)
                go.add_neighbor(&self.string[string_id], neighbor_string_id, 0)


    cdef void add_stone_to_string(self, go.string_t *string, int pos, int head):
        cdef int string_pos

        print 'AddStoneToString', string_pos

        if pos == go.STRING_END:
            return

        if string.origin > pos:
            self.string_next[pos] = string.origin
            string.origin = pos
        else:
            if head != 0:
                string_pos = head
            else:
                string_pos = string.origin

            while self.string_next[string_pos] < pos:
                string_pos = self.string_next[string_pos]

            self.string_next[pos] = self.string_next[string_pos]
            self.string_next[string_pos] = pos

        string.size += 1


    cdef void make_string(self, int pos, char color):
        cdef go.string_t *new_string
        cdef int string_id = 1
        cdef int lib_add = 0
        cdef int other = go.FLIP_COLOR(color)
        cdef int neighbor4[4], neighbor_pos
        cdef int neighbor_string_id
        cdef int i

        print 'MakeString', pos, color

        while self.string[string_id].flag:
            string_id += 1

        new_string = &self.string[string_id]

        go.fill_n_short(new_string.lib, go.string_lib_max, 0)
        go.fill_n_short(new_string.neighbor, go.max_neighbor, 0)

        new_string.lib[0] = go.LIBERTY_END
        new_string.neighbor[0] = go.NEIGHBOR_END
        new_string.libs = 0
        new_string.color = color
        new_string.origin = pos
        new_string.size = 1
        new_string.neighbors = 0

        self.string_id[pos] = string_id
        self.string_next[pos] = go.STRING_END

        go.get_neighbor4(neighbor4, pos)

        for i in range(4):
            neighbor_pos = neighbor4[i]
            if self.board[neighbor_pos] == go.S_EMPTY:
                lib_add = go.add_liberty(new_string, neighbor_pos, lib_add)
            elif self.board[neighbor_pos] == other:
                neighbor_string_id = self.string_id[neighbor_pos]
                go.add_neighbor(&self.string[neighbor_string_id], string_id, 0)
                go.add_neighbor(&self.string[string_id], neighbor_string_id, 0)

        new_string.flag = True 

    cdef int remove_string(self, go.string_t *string):
        cdef int next
        cdef int neighbor
        cdef int pos = string.origin
        cdef int remove_string_id = self.string_id[pos]
        cdef int remove_color = self.board[pos]

        print 'RemoveString', remove_string_id

        while True:
            self.board[pos] = go.S_EMPTY

            north_string_id = self.string_id[go.NORTH(pos, go.board_size)]
            west_string_id = self.string_id[go.WEST(pos)]
            east_string_id = self.string_id[go.EAST(pos)]
            south_string_id = self.string_id[go.SOUTH(pos, go.board_size)]

            if string[north_string_id].flag:
                go.add_liberty(&string[north_string_id], pos, 0)
            if string[west_string_id].flag:
                go.add_liberty(&string[west_string_id], pos, 0)
            if string[east_string_id].flag:
                go.add_liberty(&string[east_string_id], pos, 0)
            if string[south_string_id].flag:
                go.add_liberty(&string[south_string_id], pos, 0)

            next = self.string_next[pos]

            self.string_next[pos] = 0
            self.string_id[pos] = 0

            pos = next

            if pos == go.STRING_END:
                break

        neighbor = string.neighbor[0]
        while neighbor != go.NEIGHBOR_END:
            go.remove_neighbor_string(&string[neighbor], remove_string_id)
            neighbor = string.neighbor[neighbor]

        string.flag = False

        return string.size


cpdef test_playout(int n_playout=1, int move_limit=500):
    """ benchmark playout speed
    """
    cdef int i, n
    cdef char color = go.S_BLACK
    cdef list empties = []
    cdef list po_speeds = [], move_speeds = []

    import random
    import time

    go.set_board_size(19)

    gs = GameState()
    for n in range(n_playout):
        empties = range(361)
        ps = time.time()
        for i in range(move_limit):
            if not empties:
                break
            pos = random.choice(empties)
            empties.remove(pos)
            pos = pos + (2 * 19 + 2)
            ms = time.time()
            print pos, color
            gs.do_move(pos, color)
            color = go.FLIP_COLOR(color)
            move_speeds.append(time.time() - ms)
        po_speeds.append(time.time() - ps)

    print("Avg PO. {:.3f} us. ".format(np.mean(po_speeds)*1000*1000) +
          "Avg Move. {:.3f} ms. ".format(np.mean(move_speeds)*1000*1000*1000))

