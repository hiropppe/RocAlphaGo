from AlphaGo.cython import test_go

import unittest


class TestCBoard(unittest.TestCase):
    def test_set_board_size_9(self):
        test_go.test_set_board_size_9()

    def test_set_board_size_19(self):
        test_go.test_set_board_size_19()

    def test_make_string(self):
        test_go.test_make_string()
