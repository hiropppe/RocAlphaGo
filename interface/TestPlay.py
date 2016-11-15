"""Interface for AlphaGo self-play"""
from AlphaGo.go import GameState

AXIS = 'abcdefghijklmnopqrs'


class play_match(object):
    """Interface to handle play between two players."""

    def __init__(self, player1, player2, save_dir=None, size=19):
        # super(ClassName, self).__init__()
        self.player1 = player1
        self.player2 = player2
        self.state = GameState(size=size)
        self.current = self.player1
        self.opponent = self.player2
        # I Propose that GameState should take a top-level save directory,
        # then automatically generate the specific file name

    def _play(self):
        move = self.current.get_move(self.state)
        # TODO: Fix is_eye?
        self.state.do_move(move)  # Return max prob sensible legal move
        # self.state.write_to_disk()
        if len(self.state.history) > 1:
            if self.state.history[-1] is None and self.state.history[-2] is None \
                    and self.state.current_player == -1:
                end_of_game = True
            else:
                end_of_game = False
        else:
            end_of_game = False
        if not end_of_game:
            self.current, self.opponent = self.opponent, self.current
        return end_of_game
    
    def play(self, showboard=True):
        """ Play by current player """
        end_of_game = self._play()
        if showboard:
            self.showboard()
        return end_of_game

    def playeach(self, turn=1, showboard=True):
        """Play one turn, update game state, save to disk"""
        for i in xrange(turn*2):
            end_of_game = self._play()
            if end_of_game:
                break
        if showboard:
            self.showboard()
        return end_of_game

    def showboard(self):
        """
           a b c d e f g
         a . . . . . . . a     ;W(bb)
         b . o . . . . . b
         c . . . B . . . c
         d . . . . . . . d
         e . . W . B . . e
         f . . . . . . . f
         g . . . . . . . g
           a b c d e f g
        """
        board = []
        for i in xrange(self.state.size + 2):
            for j in xrange(self.state.size + 2):
                if i in (0, self.state.size + 1) and j in (0, self.state.size + 1):
                    ch = ' '
                elif i in (0, self.state.size + 1):
                    ch = AXISS[j-1]
                elif j in (0, self.state.size + 1):
                    ch = AXISS[i-1]
                elif self.state.board[j-1][i-1] == 1:
                    if self.state.history[-1][0] == j-1 and self.state.history[-1][1] == i-1:
                        ch = 'x'
                    else:
                        ch = 'B'
                elif self.state.board[j-1][i-1] == -1:
                    if self.state.history[-1][0] == j-1 and self.state.history[-1][1] == i-1:
                        ch = 'o'
                    else:
                        ch = 'W'
                elif self.state.size == 19 and (i-1 in (3, 9, 15)) and (j-1 in (3, 9, 15)):
                    ch = '+'
                else:
                    ch = '.'

                board.append(ch)
                board.append(' ')

            if i == 1 and self.state.history:
                board.append('    ;{}({}{})'.format('W' if self.state.current_player == 1 else 'B',
                                                    AXISS[self.state.history[-1][0]],
                                                    AXISS[self.state.history[-1][1]]))

            board.append('\n')

        print(''.join(board))
