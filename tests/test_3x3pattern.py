from AlphaGo.go import BLACK, WHITE
from AlphaGo.preprocessing.pattern import get_3x3_pattern

import unittest
import parseboard


class Test3x3Pattern(unittest.TestCase):
    def test_empty_onehot(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". . . . .|"
                                     ". . a . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK

        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.__assert_hot_indices(pat, [0]*8)

    def test_W1_onehot(self):
        st, moves = parseboard.parse(". B . . .|"
                                     ". W B . .|"
                                     ". B a . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 1])

    def test_W2_onehot(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". W B . .|"
                                     ". B a . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 2])

    def test_W3_onehot(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". W . . .|"
                                     ". B a . .|"
                                     ". . . W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 3])
        self.assertTrue(pat[7, 3])

    def test_B1_onehot(self):
        st, moves = parseboard.parse(". W . . .|"
                                     ". B W . .|"
                                     ". W a . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 4])

    def test_B2_onehot(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". B W . .|"
                                     ". W a . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 5])

    def test_B3_onehot(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". B . . .|"
                                     ". W a . .|"
                                     ". . . B .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.assertTrue(pat[0, 6])
        self.assertTrue(pat[7, 6])

    def test_out_of_board_onehot(self):
        st, moves = parseboard.parse(". . a . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        self.__assert_hot_indices(pat, [7, 7, 7, 0, 0, 0, 0, 0])

    def test_symmetric(self):
        # noop 
        st, moves = parseboard.parse(". . . . .|"
                                     ". B W B .|"
                                     ". W a B .|"
                                     ". W B W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        pat = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # rot90
        st, moves = parseboard.parse(". . . . .|"
                                     ". B B W .|"
                                     ". W a B .|"
                                     ". B W W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        rot90 = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # rot180
        st, moves = parseboard.parse(". . . . .|"
                                     ". W B W .|"
                                     ". B a W .|"
                                     ". B W B .|"
                                     ". . . . .|")
        st.current_player = BLACK
        rot180 = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # rot270
        st, moves = parseboard.parse(". . . . .|"
                                     ". W W B .|"
                                     ". B a W .|"
                                     ". W B B .|"
                                     ". . . . .|")
        st.current_player = BLACK
        rot270 = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # fliplr
        st, moves = parseboard.parse(". . . . .|"
                                     ". B W B .|"
                                     ". B a W .|"
                                     ". W B W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        fliplr = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # flipup
        st, moves = parseboard.parse(". . . . .|"
                                     ". W B W .|"
                                     ". W a B .|"
                                     ". B W B .|"
                                     ". . . . .|")
        st.current_player = BLACK
        flipud = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # transpose
        st, moves = parseboard.parse(". . . . .|"
                                     ". B W W .|"
                                     ". W a B .|"
                                     ". B B W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        transpose = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))
        # fliplr(rot90)
        st, moves = parseboard.parse(". . . . .|"
                                     ". W B B .|"
                                     ". B a W .|"
                                     ". W W B .|"
                                     ". . . . .|")
        st.current_player = BLACK
        fliplr_rot90 = get_3x3_pattern(st, moves['a'], symmetric=True, reverse=False).reshape((8, 8))

        hot_indices = [5, 2, 6, 3, 6, 3, 5, 2]
        self.__assert_hot_indices(pat, hot_indices)
        self.__assert_hot_indices(rot90, hot_indices)
        self.__assert_hot_indices(rot180, hot_indices)
        self.__assert_hot_indices(rot270, hot_indices)
        self.__assert_hot_indices(fliplr, hot_indices)
        self.__assert_hot_indices(flipud, hot_indices)
        self.__assert_hot_indices(transpose, hot_indices)
        self.__assert_hot_indices(fliplr_rot90, hot_indices)

    def test_reverse(self):
        st, moves = parseboard.parse(". . . . .|"
                                     ". B W B .|"
                                     ". W a B .|"
                                     ". W B W .|"
                                     ". . . . .|")
        st.current_player = BLACK
        black_pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=False).reshape((8, 8))

        st, moves = parseboard.parse(". . . . .|"
                                     ". W B W .|"
                                     ". B a W .|"
                                     ". B W B .|"
                                     ". . . . .|")
        st.current_player = WHITE
        white_pat = get_3x3_pattern(st, moves['a'], symmetric=False, reverse=True).reshape((8, 8))

        hot_indices = [5, 2, 6, 3, 6, 3, 5, 2]
        self.__assert_hot_indices(black_pat, hot_indices)
        self.__assert_hot_indices(white_pat, hot_indices)

    def __assert_hot_indices(self, pat, hot_indices):
        for i, idx in enumerate(hot_indices):
            self.assertTrue(pat[i, idx])


if __name__ == '__main__':
    unittest.main()
