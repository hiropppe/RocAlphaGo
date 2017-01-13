from AlphaGo.go import BLACK, WHITE
from AlphaGo.preprocessing.pattern import get_diamond_pattern

import unittest
import parseboard


class TestDia12Pattern(unittest.TestCase):
    def test_empty_onehot(self):
        st, moves = parseboard.parse("W W B W B W B|"
                                     "W B W . B W B|"
                                     "W B . . . W B|"
                                     "W . . W . . B|"
                                     "W B . . . W B|"
                                     "W B W . B W B|"
                                     "W B W B W B B|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, (3, 3), symmetric=False, reverse=False).reshape((12, 8))

        self.__assert_hot_indices(pat, [0]*12)

    def test_W1_onehot(self):
        st, moves = parseboard.parse(". . . B . . .|"
                                     ". . B W B . .|"
                                     ". . . . . . .|"
                                     ". . . a . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[0, 1])

    def test_W2_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . B . . . .|"
                                     ". B W . . . .|"
                                     ". . . a . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[1, 2])

    def test_W3_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . . W . . .|"
                                     ". . . a . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[2, 3])

    def test_B1_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". W . . . . .|"
                                     "W B . a . . .|"
                                     ". W . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[4, 4])

    def test_B2_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". W B a . . .|"
                                     ". . W . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[5, 5])

    def test_B3_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . a B W .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[6, 6])

    def test_out_of_board_onehot(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . a . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        self.assertTrue(pat[11, 7])

    def test_symmetric(self):
        # noop
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . W B W . .|"
                                     ". B W a B W .|"
                                     ". . B W B . .|"
                                     ". . . W . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        pat = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # rot90
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . W . . .|"
                                     ". . W B B . .|"
                                     ". B B a W W .|"
                                     ". . W W B . .|"
                                     ". . . B . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        rot90 = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # rot180
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . W . . .|"
                                     ". . B W B . .|"
                                     ". W B a W B .|"
                                     ". . W B W . .|"
                                     ". . . B . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        rot180 = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # rot270
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . B W W . .|"
                                     ". W W a B B .|"
                                     ". . B B W . .|"
                                     ". . . W . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        rot270 = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # fliplr
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . W B W . .|"
                                     ". W B a W B .|"
                                     ". . B W B . .|"
                                     ". . . W . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        fliplr = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # flipup
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . W . . .|"
                                     ". . B W B . .|"
                                     ". B W a B W .|"
                                     ". . W B W . .|"
                                     ". . . B . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        flipud = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # transpose
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . W W B . .|"
                                     ". B B a W W .|"
                                     ". . W B B . .|"
                                     ". . . W . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        transpose = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))
        # fliplr(rot90)
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . W . . .|"
                                     ". . B B W . .|"
                                     ". W W a B B .|"
                                     ". . B W W . .|"
                                     ". . . B . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        fliplr_rot90 = get_diamond_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((12, 8))

        hot_indices = [6, 5, 3, 3, 3, 3, 6, 6, 6, 6, 2, 3]
        self.__assert_hot_indices(pat, hot_indices)
        self.__assert_hot_indices(rot90, hot_indices)
        self.__assert_hot_indices(rot180, hot_indices)
        self.__assert_hot_indices(rot270, hot_indices)
        self.__assert_hot_indices(fliplr, hot_indices)
        self.__assert_hot_indices(flipud, hot_indices)
        self.__assert_hot_indices(transpose, hot_indices)
        self.__assert_hot_indices(fliplr_rot90, hot_indices)

    def test_reverse(self):
        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B . . .|"
                                     ". . W B W . .|"
                                     ". B W a B W .|"
                                     ". . B W B . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = BLACK
        black_pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((12, 8))

        st, moves = parseboard.parse(". . . . . . .|"
                                     ". . . W . . .|"
                                     ". . B W B . .|"
                                     ". W B a W B .|"
                                     ". . W B W . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        st.current_player = WHITE
        white_pat = get_diamond_pattern(st, moves['a'], symmetric=False, reverse=True).reshape((12, 8))

        hot_indices = [6, 3, 6, 2, 6, 3, 6, 3, 5, 2, 6, 0]
        self.__assert_hot_indices(black_pat, hot_indices)
        self.__assert_hot_indices(white_pat, hot_indices)

    def __assert_hot_indices(self, pat, hot_indices):
        for i, idx in enumerate(hot_indices):
            self.assertTrue(pat[i, idx])


if __name__ == '__main__':
    unittest.main()
