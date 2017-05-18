from AlphaGo.cython import test_go

import unittest


class TestCBoard(unittest.TestCase):
    def test_set_board_size_9(self):
        test_go.test_set_board_size_9()

    def test_set_board_size_19(self):
        test_go.test_set_board_size_19()

    def test_add_liberty_to_isolated_one(self):
        test_go.test_add_liberty_to_isolated_one()

    def test_remove_liberty_of_isolated_one(self):
        test_go.test_remove_liberty_of_isolated_one()

    def test_make_string_of_isolated_one(self):
        test_go.test_make_string_of_isolated_one()

    def test_add_neighbor_to_isolated_one(self):
        test_go.test_add_neighbor_to_isolated_one()

    def test_add_stone_to_string_less_position(self):
        test_go.test_add_stone_to_string_less_position()

    def test_add_stone_to_string_larger_position_from_zero_head(self):
        test_go.test_add_stone_to_string_larger_position_from_zero_head()

    def test_add_stone_to_string_larger_position_from_nonzero_head(self):
        test_go.test_add_stone_to_string_larger_position_from_nonzero_head()

    def test_add_stone_to_isolated_stone(self):
        test_go.test_add_stone_to_isolated_stone()

    def test_merge_string_two(self):
        test_go.test_merge_string_two()

    def test_merge_string_three(self):
        test_go.test_merge_string_three()

    def test_is_legal_stone_exists(self):
        test_go.test_is_legal_stone_exists()

