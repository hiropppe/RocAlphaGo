# -*- coding: utf-8 -*-

cdef extern from "ray.h":
    unsigned int REV(unsigned int p)
    unsigned int REV2(unsigned int p)
    unsigned int REV3(unsigned int p)
    unsigned int REV4(unsigned int p)
    unsigned int REV6(unsigned int p)
    unsigned int REV8(unsigned int p)
    unsigned int REV10(unsigned int p)
    unsigned int REV12(unsigned int p)
    unsigned int REV14(unsigned int p)
    unsigned int REV16(unsigned int p)
    unsigned int REV18(unsigned int p)

    int MD2_MAX
    int PAT3_MAX
    int MD2_LIMIT
    int PAT3_LIMIT

    ctypedef enum MD:
        MD_2
        MD_3
        MD_4
        MD_MAX

    ctypedef enum LARGE_MD:
        MD_5
        MD_LARGE_MAX

    ctypedef struct pattern_t:
        unsigned int *list
        unsigned long long *large_list


cdef int N
cdef int S
cdef int W
cdef int E
cdef int NN
cdef int NW
cdef int NE
cdef int SS
cdef int SW
cdef int SE
cdef int WW
cdef int EE

# init
cdef void clear_pattern(pattern_t *pat)

# update
cdef void update_md2_stone(pattern_t *pat, int pos, char color)
cdef void update_md2_empty(pattern_t *pat, int pos)

# symmetric 
cdef void pat3_transpose8(unsigned int pat3, unsigned int *transp)
cdef void pat3_transpose16(unsigned int pat3, unsigned int *transp)

# flip color
cdef unsigned int pat3_reverse(unsigned int pat3)

# vertical mirror
cdef unsigned int pat3_vertical_mirror(unsigned int pat3)

# horizontal mirror
cdef unsigned int pat3_horizontal_mirror(unsigned int pat3)

# rotate
cdef unsigned int pat3_rotate90(unsigned int pat3)

# search pattern
cdef unsigned int pat3(pattern_t *pat, int pos)

# print pattern
cdef void print_input_pat3(unsigned int pat3)
