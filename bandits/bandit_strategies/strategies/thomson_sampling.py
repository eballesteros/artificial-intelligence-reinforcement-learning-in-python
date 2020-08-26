import numpy as np

from .greedy import Greedy
from ..bandit import Bandit

class BayesianBandit(Bandit):
    def __init__(self, true_p, alpha_0=1, beta_0=1):
        '''
        Bayesian version of the Bandit.

        Posterior ~ Bernoulli(true_p)
        Default prior = Non-informative prions = Beta(1,1) = Uniform[0,1]

        :param true_p:
        :param alpha_0:
        :param beta_0:
        '''
        super().__init__(true_p)

        self._alpha = alpha_0
        self._beta = beta_0

    def sample(self):
        return np.random.beta(self._alpha, self._beta)

    def _update(self, x):
        super()._update(x)

        self._alpha += x
        self._beta += 1 - x

class ThomsonSampling(Greedy):
    def __init__(self, alpha_0=1, beta_0=1):
        self._alpha = alpha_0
        self._beta = beta_0

    def __str__(self):
        return 'ThomsonSampling'

    def _pick_best_bandit(self, bandit_list, *args, **kargs):
        return bandit_list[np.argmax([b.sample() for b in bandit_list])]

    def init_bandit(self, true_p):
        return BayesianBandit(true_p, alpha_0=self._alpha, beta_0=self._beta)


    