import numpy as np

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

    def test_sample(self):
        st, moves = parseboard.parse("B . . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|"
                                     ". . . . .|")
        st.current_player = WHITE

        self.rf.pattern_3x3 = {13222222220000000: 0}
        self.rf.update_non_response_3x3(st, self.F)

        self.assertTrue(self.F[6, self.rf.ix_3x3 + 0])


if __name__ == '__main__':
    unittest.main()
