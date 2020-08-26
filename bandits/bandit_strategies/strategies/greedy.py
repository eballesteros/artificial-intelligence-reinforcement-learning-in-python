import numpy as np

from .strategy import Strategy
from ..bandit import Bandit

class Greedy(Strategy):
    def __str__(self):
        return 'Greedy'

    @staticmethod
    def _pick_best_bandit(bandit_list):
        return bandit_list[np.argmax([b.p_estimate for b in bandit_list])]

    def choose(self, bandit_list):
        return self._pick_best_bandit(bandit_list), False

    def init_bandit(true_p):
        return Bandit(true_p, initial_p_estimate=0)