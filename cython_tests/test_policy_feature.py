from AlphaGo.cython import test_policy_feature as ctest

import unittest


class TestPolicyFeature(unittest.TestCase):
    def test_stone_color(self):
        ctest.test_stone_color()

    def test_turns_since(self):
        ctest.test_turns_since()

    def test_liberties(self):
        ctest.test_liberties()

    def test_capture_size(self):
        ctest.test_capture_size()

    def test_self_atari_size(self):
        ctest.test_self_atari_size()

    def test_liberties_after_move_1(self):
        ctest.test_liberties_after_move_1()

    def test_sensibleness(self):
        ctest.test_sensibleness()
