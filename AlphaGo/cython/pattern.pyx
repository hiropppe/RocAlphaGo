# -*- coding: utf-8 -*-

cdef void clear_pattern(pattern_t *pat):
    pass


cdef void pat3_transpose8(unsigned int pat3, unsigned int *transp):
    transp[0] = pat3;
    transp[1] = pat3_vertical_mirror(pat3);
    transp[2] = pat3_horizontal_mirror(pat3);
    transp[3] = pat3_vertical_mirror(transp[2]);
    transp[4] = pat3_rotate90(pat3);
    transp[5] = pat3_rotate90(transp[1]);
    transp[6] = pat3_rotate90(transp[2]);
    transp[7] = pat3_rotate90(transp[3]);


cdef void pat3_transpose16(unsigned int pat3, unsigned int *transp):
    transp[0] = pat3
    transp[1] = pat3_vertical_mirror(pat3)
    transp[2] = pat3_horizontal_mirror(pat3)
    transp[3] = pat3_vertical_mirror(transp[2])
    transp[4] = pat3_rotate90(pat3);
    transp[5] = pat3_rotate90(transp[1])
    transp[6] = pat3_rotate90(transp[2])
    transp[7] = pat3_rotate90(transp[3])
    transp[8] = pat3_reverse(transp[0])
    transp[9] = pat3_reverse(transp[1])
    transp[10] = pat3_reverse(transp[2])
    transp[11] = pat3_reverse(transp[3])
    transp[12] = pat3_reverse(transp[4])
    transp[13] = pat3_reverse(transp[5])
    transp[14] = pat3_reverse(transp[6])
    transp[15] = pat3_reverse(transp[7])


cdef unsigned int pat3_reverse(unsigned int pat3):
    return ((pat3 >> 1) & 0x5555) | ((pat3 & 0x5555) << 1)


cdef unsigned int pat3_vertical_mirror(unsigned int pat3):
    return ((pat3 & 0xFC00) >> 10) | (pat3 & 0x03C0) | ((pat3 & 0x003F) << 10)


cdef unsigned int pat3_horizontal_mirror(unsigned int pat3):
    return (REV3((pat3 & 0xFC00) >> 10) << 10) | (REV((pat3 & 0x03C0) >> 6) << 6) | (REV3((pat3 & 0x003F)))


cdef unsigned int pat3_rotate90(unsigned int pat3):
    """
    1 2 3    3 5 8
    4   5 -> 2   7
    6 7 8    1 4 6
    """
    return ((pat3 & 0x0003) << 10) | ((pat3 & 0x0C0C) << 4) | ((pat3 & 0x3030) >> 4) | ((pat3 & 0x00C0) << 6) | ((pat3 & 0x0300) >> 6) | ((pat3 & 0xC000) >> 10)
