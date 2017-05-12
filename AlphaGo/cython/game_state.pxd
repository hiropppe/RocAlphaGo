# -*- coding: utf-8 -*-

cimport go


cdef class GameState:
    cdef:
        char *board

        go.string_t *string
        int *string_id
        int *string_next

        readonly int moves
        go.move_t *record

        int *prisoner

        readonly int ko_pos
        readonly int ko_move

        readonly int pass_count

        readonly unsigned long long current_hash

    cdef void do_move(self, int pos, char color)
    
    cdef void connect_string(self, int pos, char color, int connection, int string_id[4])

    cdef void merge_string(self, go.string_t *dst, go.string_t *src[3], int n)

    cdef void add_stone(self, int pos, char color, int string_id)

    cdef void add_stone_to_string(self, go.string_t *string, int pos, int head)

    cdef void make_string(self, int pos, char color)

    cdef int remove_string(self, go.string_t *string)
