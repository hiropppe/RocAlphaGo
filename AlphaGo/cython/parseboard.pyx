cimport go


cdef tuple parse(go.game_state_t *game, boardstr):
    '''Parses a board into a gamestate, and returns the location of any moves
    marked with anything other than 'B', 'X', '#', 'W', 'O', or '.'

    Rows are separated by '|', spaces are ignored.

    '''
    cdef int row, col, pos
    cdef dict moves = {}
    cdef dict pure_moves = {}

    boardstr = boardstr.replace(' ', '')
    board_size = max(boardstr.index('|'), boardstr.count('|'))

    # assert board_size in (7, 9, 13, 19), "Illegal board size"
    go.set_board_size(board_size)

    go.initialize_board(game, False)

    for row, rowstr in enumerate(boardstr.split('|')):
        for col, c in enumerate(rowstr):
            pos = go.POS(col + go.OB_SIZE, row + go.OB_SIZE, go.board_size)
            if c == '.':
                continue  # ignore empty spaces
            elif c in 'BX#':
                go.put_stone(game, pos, color=go.S_BLACK)
            elif c in 'WO':
                go.put_stone(game, pos, color=go.S_WHITE)
            else:
                # move reference
                assert c not in moves, "{} already used as a move marker".format(c)
                moves[c] = go.POS(col + go.OB_SIZE, row + go.OB_SIZE, go.board_size)
                pure_moves[c] = go.POS(col, row, board_size)

    return moves, pure_moves
