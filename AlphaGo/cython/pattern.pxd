# -*- coding: utf-8 -*-

# symmetric 
cdef void pat3_transpose8(unsigned int pat3, unsigned int *transp)
cdef void pat3_transpose16(unsigned int pat3, unsigned int *transp)

cdef extern from "go_pattern.h":
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

    unsigned char *eye
    unsigned char *false_eye
    unsigned char *territory
    unsigned char *nb4_empty
    unsigned char *eye_condition

    ctypedef enum MD:
        MD_2
        MD_3
        MD_4
        MD_MAX

    ctypedef enum LARGE_MD:
        MD_5
        MD_LARGE_MAX

    ctypedef enum eye_condition:
        E_NOT_EYE
        E_COMPLETE_HALF_EYE
        E_HALF_3_EYE
        E_HALF_2_EYE
        E_HALF_1_EYE
        E_COMPLETE_ONE_EYE
        E_MAX

    ctypedef struct pattern_t:
        unsigned int *list
        unsigned long long *large_list


cdef void clear_pattern(pattern_t *pat)


# flip color
cdef unsigned int pat3_reverse(unsigned int pat3)

# vertical mirror
cdef unsigned int pat3_vertical_mirror(unsigned int pat3)

# horizontal mirror
cdef unsigned int pat3_horizontal_mirror(unsigned int pat3)

# rotate
cdef unsigned int pat3_rotate90(unsigned int pat3)

