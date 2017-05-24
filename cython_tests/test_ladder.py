from AlphaGo.cython import test_ladder

import unittest


class TestLadder(unittest.TestCase):
    def test_captured_1(self):
        test_ladder.test_captured_1()

    def test_breaker_1(self):
        test_ladder.test_breaker_1()

    def test_missing_ladder_breaker_1(self):
        test_ladder.test_missing_ladder_breaker_1()

    def test_capture_to_escape_1(self):
        test_ladder.test_capture_to_escape_1()

    def test_throw_in_1(self):
        test_ladder.test_throw_in_1()

    def test_snapback_1(self):
        test_ladder.test_snapback_1()

    def test_two_captures(self):
        test_ladder.test_two_captures()

    def test_two_escapes(self):
        test_ladder.test_two_escapes()
