from AlphaGo.go import BLACK, WHITE
from AlphaGo.preprocessing import nakade

import unittest
import parseboard


class TestNakade(unittest.TestCase):

    @classmethod
    def setup_class(clazz):
        nakade.initialize_hash()

    def test_nakade3_0(self):
        board = \
        ". B B B B B . . . . . . . . . . . . .|" + \
        "B W W W W W B . . . . . . . . . . . .|" + \
        "B W . . . W B . . . . . . . . . . . .|" + \
        "B W W . W W B . . . . . . . . . . . .|" + \
        ". B B B B B . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|"

        st, moves = parseboard.parse(board)
        st.current_player = WHITE
        st.do_move((3, 3))
        nakade.search_nakade(st)

    def test_nakade4_0(self):
        board = \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . B B B B B . . . . . . .|" + \
        ". . . . . . . B W W W B . . . . . . .|" + \
        ". . . . . . B B W . W B B . . . . . .|" + \
        ". . . . . . B W . . . W B . . . . . .|" + \
        ". . . . . . B W W . W W B . . . . . .|" + \
        ". . . . . . B B B B B B B . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|" + \
        ". . . . . . . . . . . . . . . . . . .|"

        st, moves = parseboard.parse(board)
        st.current_player = WHITE
        st.do_move((11, 9))
        nakade.search_nakade(st)


if __name__ == '__main__':
    unittest.main()
