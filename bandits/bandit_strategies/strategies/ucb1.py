import numpy as np

from .greedy import Greedy

class UCB1(Greedy):
    def __init__(self, trial_n_offset=2):
        '''
        :param trial_n_offset: trial_n goes in a log, need to offset so its never log(0) or log(1)
        '''
        self.trial_n_offset = trial_n_offset

    def __str__(self):
        return 'UCB1'

    def init_bandit(self, true_p):
        bandit = super().init_bandit(true_p)
        _ = bandit.pull() #pull once during init
        return bandit

    def _upper_error_bound(self, bandit, trial_n):
        return np.sqrt(2 * np.log(trial_n + self.trial_n_offset) / bandit.n_pulls)

    def _pick_best_bandit(self, bandit_list, trial_n):
        return bandit_list[np.argmax([b.p_estimate + self._upper_error_bound(b, trial_n) for b in bandit_list])]