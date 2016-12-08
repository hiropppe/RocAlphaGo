import math
import time


class Canadian(object):
    """ A simple consideration time implementation based on Canadian time settings.
        Just calculate time per stone from remaining time and stones (time remaining
        / stone remaining).
        Remaining stone for main time is given in a fixed number(50).
    """

    def __init__(self, m=2000, ms=50, b=180, bs=6):
        self.m = m
        self.ms = ms
        self.b = b
        self.bs = bs

        self.total_elapsed = 0
        self.start_time = 0

        # Initialize with fixed number of stone (50) for main time.
        # If main time not set, byo-yomi time is set instead.
        if m:
            self.time_left = m
            self.stone_left = ms
        else:
            self.time_left = b
            self.stone_left = bs

        # Consider .5 sec overhead
        self.overhead = .5

    def start(self, state=None):
        self.start_time = time.time()
        cons_time = self.time_left/self.stone_left
        return max(1, math.ceil(cons_time - self.overhead))

    def stop(self):
        elapsed = time.time() - self.start_time
        self.total_elapsed += elapsed
        self.time_left -= elapsed
        self.stone_left -= 1

        if 0 < self.time_left:
            if self.stone_left == 0:
                if self.total_elapsed < self.m:
                    self.stone_left = 1
                else:
                    self.time_left = self.b
                    self.stone_left = self.bs
        else:
            # Never lose by timeout
            self.time_left = self.b
            self.stone_left = self.bs

    def set_time_left(self, time_left, stone_left):
        self.b = time_left
        self.time_left = time_left
        if 0 < stone_left:
            self.bs = stone_left
            self.stone_left = stone_left
