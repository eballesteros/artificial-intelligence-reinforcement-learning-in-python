import numpy as np

class Bandit:
    def __init__(self, true_p, initial_p_estimate=0):
        '''
        :param true_p: float; chance of getting 1 when pulling (0 to 1)
        :param initial_p_estimate: float; initial estimate for p (default 0)
        '''
        assert 0. <= true_p <= 1.
        self._true_p = true_p

        self.p_estimate = initial_p_estimate
        self.n_pulls = 1 if initial_p_estimate else 0 #in initial_p_estimate and 0, first update erases initial estimate

    def _update(self, x):
        '''
        update internal estimates
        '''
        self.n_pulls += 1
        self.p_estimate += (1/self.n_pulls) * (x - self.p_estimate)

    def pull(self):
        '''
        :returns x: 1 with probability true_p, else 0
        '''
        x = int(np.random.rand() < self._true_p)
        self._update(x)
        return x