import numpy as np

import AlphaGo.go as go

from AlphaGo.go import BLACK, WHITE
from AlphaGo.preprocessing import rollout_feature

import unittest
import parseboard


class TestNonResponse3x3(unittest.TestCase):

    def setUp(self):
        self.rf = rollout_feature.RolloutFeature(5)
        self.rf.pattern_3x3 = {}
        self.F = np.zeros((self.rf.n_position, self.rf.n_feature), dtype=np.int)

    def test_initial_board(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK

        self.rf.update_non_response_3x3(st, self.F)

        self.assertTrue(self.F.sum() == 0)

    def test_pattern_for_black(self):
        st, moves = parseboard.parse(". W . . .|"
                                     ". B W B .|"
                                     ". W a . .|"
                                     ". W . W .|"
                                     ". . . . .|")
        st.current_player = BLACK

        self.rf.pattern_3x3 = {33131212112330303: 0}
        self.rf.update_non_response_3x3(st, self.F)

        self.assertTrue(self.F.sum() == 1)
        self.assertTrue(self.F[self.__move_to_position(moves['a'], st.size), self.rf.ix_3x3 + 0])

    def test_pattern_for_white(self):
        st, moves = parseboard.parse(". W . . .|"
                                     ". B W B .|"
                                     ". W a . .|"
                                     ". W . W .|"
                                     ". . . . .|")
        st.current_player = WHITE

        self.rf.pattern_3x3 = {13131212112330303: 0}
        self.rf.update_non_response_3x3(st, self.F)

        self.assertTrue(self.F.sum() == 1)
        self.assertTrue(self.F[self.__move_to_position(moves['a'], st.size), self.rf.ix_3x3 + 0])

    def test_get_updated_positions(self):
        st = go.GameState(5)

        diff = self.rf.get_updated_positions(st)
        self.assertTrue(len(diff) == 0)

        st.do_move((2, 2))
        diff = self.rf.get_updated_positions(st)
        self.assertTrue(len(diff) == 1)
        self.assertEquals(self.__move_to_position((2, 2), st.size), diff[0])

    def test_get_updated_positions_atari(self):
        st, moves = parseboard.parse("B B a . .|"
                                     "W W . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK

        self.rf.get_updated_positions(st)

        st.do_move(moves['a'])
        diff = self.rf.get_updated_positions(st)

        self.assertTrue(len(diff) == 3)
        self.assertTrue(self.__move_to_position((0, 0), st.size) in diff)
        self.assertTrue(self.__move_to_position((0, 1), st.size) in diff)
        self.assertTrue(self.__move_to_position((0, 2), st.size) in diff)

    def test_get_updated_positions_group(self):
        st, moves = parseboard.parse("B B a . .|"
                                     ". W . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK

        self.rf.get_updated_positions(st)

        st.do_move(moves['a'])
        diff = self.rf.get_updated_positions(st)

        self.assertTrue(len(diff) == 3)
        self.assertTrue(self.__move_to_position((0, 0), st.size) in diff)
        self.assertTrue(self.__move_to_position((0, 1), st.size) in diff)
        self.assertTrue(self.__move_to_position((0, 2), st.size) in diff)

    def test_ignore_pattern_including_out_of_board(self):
        st, moves = parseboard.parse("B a W . .|"
                                     "b W B . .|"
                                     "W B . W B|"
                                     ". . W B c|"
                                     ". . B d W|")
        st.current_player = BLACK

        self.assertEquals(self.rf.get_3x3_key(st, moves['a']), -1)
        self.assertEquals(self.rf.get_3x3_key(st, moves['b']), -1)
        self.assertEquals(self.rf.get_3x3_key(st, moves['c']), -1)
        self.assertEquals(self.rf.get_3x3_key(st, moves['d']), -1)

    def __move_to_position(self, move, board_size):
        (x, y) = move
        return x * board_size + y


if __name__ == '__main__':
    unittest.main()
