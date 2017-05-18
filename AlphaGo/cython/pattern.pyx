# -*- coding: utf-8 -*-

from libc.string cimport memset

cimport go


cdef unsigned int update_mask[40][3]

update_mask[:] = [
  # 3x3
  [ 0, 0x00004000, 0x00008000 ], #  1->8
  [ 0, 0x00001000, 0x00002000 ], #  2->7
  [ 0, 0x00000400, 0x00000800 ], #  3->6
  [ 0, 0x00000100, 0x00000200 ], #  4->5
  [ 0, 0x00000040, 0x00000080 ], #  5->4
  [ 0, 0x00000010, 0x00000020 ], #  6->3
  [ 0, 0x00000004, 0x00000008 ], #  7->2
  [ 0, 0x00000001, 0x00000002 ], #  8->1
  # md2
  [ 0, 0x00100000, 0x00200000 ], #  9->11
  [ 0, 0x00400000, 0x00800000 ], # 10->12
  [ 0, 0x00010000, 0x00020000 ], # 11-> 9
  [ 0, 0x00040000, 0x00080000 ], # 12->10
  # md3
  [ 0, 0x00001000, 0x00002000 ], # 13->19
  [ 0, 0x00004000, 0x00008000 ], # 14->20
  [ 0, 0x00010000, 0x00020000 ], # 15->21
  [ 0, 0x00040000, 0x00080000 ], # 16->22
  [ 0, 0x00100000, 0x00200000 ], # 17->23
  [ 0, 0x00400000, 0x00800000 ], # 18->24
  [ 0, 0x00000001, 0x00000002 ], # 19->13
  [ 0, 0x00000004, 0x00000008 ], # 20->14
  [ 0, 0x00000010, 0x00000020 ], # 21->15
  [ 0, 0x00000040, 0x00000080 ], # 22->16
  [ 0, 0x00000100, 0x00000200 ], # 23->17
  [ 0, 0x00000400, 0x00000800 ], # 24->18
  # md4 
  [ 0, 0x00010000, 0x00020000 ], # 25->33
  [ 0, 0x00040000, 0x00080000 ], # 26->34
  [ 0, 0x00100000, 0x00200000 ], # 27->35
  [ 0, 0x00400000, 0x00800000 ], # 28->36
  [ 0, 0x01000000, 0x02000000 ], # 29->37
  [ 0, 0x04000000, 0x08000000 ], # 30->38
  [ 0, 0x10000000, 0x20000000 ], # 31->39
  [ 0, 0x40000000, 0x80000000 ], # 32->40
  [ 0, 0x00000001, 0x00000002 ], # 33->25
  [ 0, 0x00000004, 0x00000008 ], # 34->26
  [ 0, 0x00000010, 0x00000020 ], # 35->27
  [ 0, 0x00000040, 0x00000080 ], # 36->28
  [ 0, 0x00000100, 0x00000200 ], # 37->29
  [ 0, 0x00000400, 0x00000800 ], # 38->30
  [ 0, 0x00001000, 0x00002000 ], # 39->31
  [ 0, 0x00004000, 0x00008000 ], # 40->32
]


cdef void clear_pattern(pattern_t *pat):
    cdef int y

    memset(pat, 0, sizeof(pattern_t) * go.board_max)

    for y in range(go.board_start, go.board_end + 1):
      # 1線
      # 上
      pat[go.POS(y, go.board_start, go.board_size)].list[<int>MD_2] |= 0x0003003F;  # 1 2 3 9
#      pat[POS(y, go.board_start)].list[MD_3] |= 0x00F0003F;  # 13 14 15 23 24
#      pat[POS(y, go.board_start)].list[MD_4] |= 0xFC0000FF;  # 25 26 27 28 38 39 40
#      pat[POS(y, go.board_start)].large_list[MD_5] |= 0xFF000003FF;
      # 右
      pat[go.POS(go.board_end, y, go.board_size)].list[<int>MD_2] |= 0x000CC330;    # 3 5 8 10
#      pat[POS(go.board_end, y)].list[MD_3] |= 0x00000FFC;    # 14 15 16 17 18
#      pat[POS(go.board_end, y)].list[MD_4] |= 0x0000FFFC;    # 26 27 28 29 30 31 32
#      pat[POS(go.board_end, y)].large_list[MD_5] |= 0x00000FFFFC;
      # 下
      pat[go.POS(y, go.board_end, go.board_size)].list[<int>MD_2] |= 0x0030FC00;    # 6 7 8 11
#      pat[POS(y, go.board_end)].list[MD_3] |= 0x0003FF00;    # 17 18 19 20 21
#      pat[POS(y, go.board_end)].list[MD_4] |= 0x00FFFC00;    # 30 31 32 33 34 35 36
#      pat[POS(y, go.board_end)].large_list[MD_5] |= 0x003FFFF000;
      # 左
      pat[go.POS(go.board_start, y, go.board_size)].list[<int>MD_2] |= 0x00C00CC3;  # 1 4 6 12
#      pat[POS(go.board_start, y)].list[MD_3] |= 0x00FFC000;  # 20 21 22 23 24
#      pat[POS(go.board_start, y)].list[MD_4] |= 0xFFFC0000;  # 34 35 36 37 38 39 40
#      pat[POS(go.board_start, y)].large_list[MD_5] |= 0xFFFFC00000;

      # 2線
      # 上
      pat[go.POS(y, go.board_start + 1, go.board_size)].list[<int>MD_2] |= 0x00030000;      # 9
#      pat[POS(y, go.board_start + 1)].list[MD_3] |= 0x00C0000F;      # 13 14 24
#      pat[POS(y, go.board_start + 1)].list[MD_4] |= 0xF000003F;      # 25 26 27 39 40
#      pat[POS(y, go.board_start + 1)].large_list[MD_5] |= 0xFC000000FF;
      # 右
      pat[go.POS(go.board_end - 1, y, go.board_size)].list[<int>MD_2] |= 0x000C0000;        # 10
#      pat[POS(go.board_end - 1, y)].list[MD_3] |= 0x000003F0;        # 15 16 17
#      pat[POS(go.board_end - 1, y)].list[MD_4] |= 0x00003FF0;        # 27 28 29 30 31
#      pat[POS(go.board_end - 1, y)].large_list[MD_5] |= 0x000003FFF0;
      # 下
      pat[go.POS(y, go.board_end - 1, go.board_size)].list[<int>MD_2] |= 0x00300000;        # 11
#      pat[POS(y, go.board_end - 1)].list[MD_3] |= 0x0000FC00;        # 18 19 20
#      pat[POS(y, go.board_end - 1)].list[MD_4] |= 0x003FF000;        # 31 32 33 34 35
#      pat[POS(y, go.board_end - 1)].large_list[MD_5] |= 0x000FFFC000;
      # 左
      pat[go.POS(go.board_start + 1, y, go.board_size)].list[<int>MD_2] |= 0x00C00000;      # 12
#      pat[POS(go.board_start + 1, y)].list[MD_3] |= 0x003F0000;      # 21 22 23
#      pat[POS(go.board_start + 1, y)].list[MD_4] |= 0x3FF00000;      # 35 36 37 38 39
#      pat[POS(go.board_start + 1, y)].large_list[MD_5] |= 0x3FFF000000;

      # 3線
      # 上
#      pat[POS(y, go.board_start + 2)].list[MD_3] |= 0x00000003;      # 13
#      pat[POS(y, go.board_start + 2)].list[MD_4] |= 0xC000000F;      # 25 26 40
#      pat[POS(y, go.board_start + 2)].large_list[MD_5] |= 0xF00000003F;
      # 右
#      pat[POS(go.board_end - 2, y)].list[MD_3] |= 0x000000C0;        # 16
#      pat[POS(go.board_end - 2, y)].list[MD_4] |= 0x00000FC0;        # 28 29 30
#      pat[POS(go.board_end - 2, y)].large_list[MD_5] |= 0x000000FFC0;
      # 下
#      pat[POS(y, go.board_end - 2)].list[MD_3] |= 0x00003000;        # 19
#      pat[POS(y, go.board_end - 2)].list[MD_4] |= 0x000FC000;        # 32 33 34
#      pat[POS(y, go.board_end - 2)].large_list[MD_5] |= 0x0003FF0000;
      # 左
#      pat[POS(go.board_start + 2, y)].list[MD_3] |= 0x000C0000;      # 22
#      pat[POS(go.board_start + 2, y)].list[MD_4] |= 0x0FC00000;      # 36 37 38
#      pat[POS(go.board_start + 2, y)].large_list[MD_5] |= 0x0FFC000000;

      # 4線
      # 上
#      pat[POS(y, go.board_start + 3)].list[MD_4] |= 0x00000003;      # 25
#      pat[POS(y, go.board_start + 3)].large_list[MD_5] |= 0xC00000000F;
      # 右
#      pat[POS(go.board_end - 3, y)].list[MD_4] |= 0x00000300;        # 29
#      pat[POS(go.board_end - 3, y)].large_list[MD_5] |= 0x0000003F00;
      # 下
#      pat[POS(y, go.board_end - 3)].list[MD_4] |= 0x00030000;        # 33
#      pat[POS(y, go.board_end - 3)].large_list[MD_5] |= 0x0000FC0000;
      # 左
#      pat[POS(go.board_start + 3, y)].list[MD_4] |= 0x03000000;      # 37
#      pat[POS(go.board_start + 3, y)].large_list[MD_5] |= 0x03F0000000;

      # 5線
      # 上
#      pat[POS(y, go.board_start + 4)].large_list[MD_5] |= 0x0000000003;
      # 右
#      pat[POS(go.board_end - 4, y)].large_list[MD_5] |= 0x0000000C00;
      # 下
#      pat[POS(y, go.board_end - 4)].large_list[MD_5] |= 0x0000300000;
      # 左
#      pat[POS(go.board_start + 4, y)].large_list[MD_5] |= 0x00C0000000;


cdef void update_md2_stone(pattern_t *pat, int pos, char color):
    pat[pos + NW].list[<int>MD_2] |= update_mask[0][<int>color];
    pat[pos +  N].list[<int>MD_2] |= update_mask[1][<int>color];
    pat[pos + NE].list[<int>MD_2] |= update_mask[2][<int>color];
    pat[pos +  W].list[<int>MD_2] |= update_mask[3][<int>color];
    pat[pos +  E].list[<int>MD_2] |= update_mask[4][<int>color];
    pat[pos + SW].list[<int>MD_2] |= update_mask[5][<int>color];
    pat[pos +  S].list[<int>MD_2] |= update_mask[6][<int>color];
    pat[pos + SE].list[<int>MD_2] |= update_mask[7][<int>color];
    pat[pos + NN].list[<int>MD_2] |= update_mask[8][<int>color];
    pat[pos + EE].list[<int>MD_2] |= update_mask[9][<int>color];
    pat[pos + SS].list[<int>MD_2] |= update_mask[10][<int>color];
    pat[pos + WW].list[<int>MD_2] |= update_mask[11][<int>color];


cdef void update_md2_empty(pattern_t *pat, int pos):
    pat[pos + NW].list[<int>MD_2] &= 0xFF3FFF;
    pat[pos +  N].list[<int>MD_2] &= 0xFFCFFF;
    pat[pos + NE].list[<int>MD_2] &= 0xFFF3FF;
    pat[pos +  W].list[<int>MD_2] &= 0xFFFCFF;
    pat[pos +  E].list[<int>MD_2] &= 0xFFFF3F;
    pat[pos + SW].list[<int>MD_2] &= 0xFFFFCF;
    pat[pos +  S].list[<int>MD_2] &= 0xFFFFF3;
    pat[pos + SE].list[<int>MD_2] &= 0xFFFFFC;
    pat[pos + NN].list[<int>MD_2] &= 0xCFFFFF;
    pat[pos + EE].list[<int>MD_2] &= 0x3FFFFF;
    pat[pos + SS].list[<int>MD_2] &= 0xFCFFFF;
    pat[pos + WW].list[<int>MD_2] &= 0xF3FFFF;


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


cdef unsigned int pat3(pattern_t *pat, int pos):
    return (pat[pos].list[<int>MD_2] & 0xFFFF)
