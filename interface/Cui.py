"""Interface for AlphaGo self-play"""
import numpy as np
import sys

from AlphaGo.go import GameState


WHITE = -1
BLACK = +1
EMPTY = 0

AXIS = 'abcdefghijklmnopqrs'
RESULT = 'DBW'


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
        if not self.state.is_end_of_game:
            self.current, self.opponent = self.opponent, self.current

        return self.state.is_end_of_game

    def clear(self, showboard=True):
        self.state = GameState(size=self.state.size)

        if showboard:
            self.showboard()

    def play(self, showboard=True):
        """ Play by current player """
        if not self.state.is_end_of_game:
            self._play()

        if showboard:
            self.showboard()

        return self.state.is_end_of_game

    def playeach(self, turn=1, showboard=True):
        """ Play each other """
        if self.state.is_end_of_game:
            return True

        for i in xrange(turn*2):
            self._play()
            if self.state.is_end_of_game:
                break

        if showboard:
            self.showboard()

        return self.state.is_end_of_game

    def playout(self, limit=500, showboard=True):
        """ Playout """
        if self.state.is_end_of_game:
            return True

        for i in range(limit):
            self._play()
            if self.state.is_end_of_game:
                break
        else:
            sys.stderr.write('{}\n'.format('WARNING: rollout reached move limit'))

        if showboard:
            self.showboard()

        return self.state.is_end_of_game

    def showboard(self):
        """
           a b c d e f g
         a . . . . . . . a     ;W(bb)
         b . W . . . . . b
         c . . . x . . . c
         d . . . . . . . d
         e . . o . x . . e
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
                    ch = AXIS[j-1]
                elif j in (0, self.state.size + 1):
                    ch = AXIS[i-1]
                elif self.state.board[j-1][i-1] == 1:
                    if self.state.history[-1] \
                       and self.state.history[-1][0] == j-1 \
                       and self.state.history[-1][1] == i-1:
                        ch = 'B'
                    else:
                        ch = 'x'
                elif self.state.board[j-1][i-1] == -1:
                    if self.state.history[-1] \
                       and self.state.history[-1][0] == j-1 \
                       and self.state.history[-1][1] == i-1:
                        ch = 'W'
                    else:
                        ch = 'o'
                elif self.state.size == 19 and (i-1 in (3, 9, 15)) and (j-1 in (3, 9, 15)):
                    ch = '+'
                else:
                    ch = '.'

                board.append(ch)
                board.append(' ')

            if i == 1 and self.state.history:
                if self.state.history[-1]:
                    board.append('    ;{}({}{})'.format('W' if self.state.current_player == 1 and not self.state.is_end_of_game else 'B',
                                                        AXIS[self.state.history[-1][0]],
                                                        AXIS[self.state.history[-1][1]]))
                else:
                    board.append('    ;{}(tt)'.format('W' if self.state.current_player == 1 and not self.state.is_end_of_game else 'B'))

            if i == 3 and self.state.is_end_of_game:
                score_white, score_black = self.calculate_score()
                board.append('    ' + ('Draw' if not self.state.get_winner() else 'Winner: {}'.format(RESULT[self.state.get_winner()])))
                board.append(' (W: {}, B: {})'.format(score_white, score_black))

            board.append('\n')

        print(''.join(board))

    def calculate_score(self):
        """Calculate score of board state and return player ID (1, -1, or 0 for tie)"""
        # Count number of positions filled by each player, plus 1 for each eye-ish space owned
        score_white = np.sum(self.state.board == WHITE)
        score_black = np.sum(self.state.board == BLACK)
        empties = zip(*np.where(self.state.board == EMPTY))
        for empty in empties:
            # Check that all surrounding points are of one color
            if self.state.is_eyeish(empty, BLACK):
                score_black += 1
            elif self.state.is_eyeish(empty, WHITE):
                score_white += 1
        score_white += self.state.komi
        score_white -= self.state.passes_white
        score_black -= self.state.passes_black
        return score_white, score_black
