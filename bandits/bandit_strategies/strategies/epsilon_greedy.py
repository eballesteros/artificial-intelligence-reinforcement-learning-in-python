import numpy as np

from .greedy import Greedy

class EpsilonGreedy(Greedy):
    def __init__(self, epsilon):
        '''
        :param epsilon: float; Chance of exploring (0 to 1)
        '''
        assert 0 <= epsilon <= 1
        self._epsilon = epsilon

    def __str__(self):
        return f'EpsilonGreedy({self._epsilon})'

    @staticmethod
    def _pick_random_bandit(bandit_list):
        return np.random.choice(bandit_list)

    def choose(self, bandit_list, *args, **kargs):
        if np.random.rand() < self._epsilon: #explore
            return self._pick_random_bandit(bandit_list), True
        else: #exploit
            return self._pick_best_bandit(bandit_list), False


class DecayingEpsilonGreedy(Greedy):
    def __init__(self, initial_epsilon, decay_factor):
        '''
        Exponentially decaying EpsilonGreedy

        :param initial_epsilon: float; Initial chance of exploring (0 to 1)
        :param decay_factor: float; Exponential decay factor
        '''
        assert 0 <= initial_epsilon <= 1
        assert 0 <= decay_factor <= 1
        self._initial_epsilon = initial_epsilon
        self._decay_factor = decay_factor
        self._time_steps = 0

    def __str__(self):
        return f'DecayingEpsilonGreedy({self._initial_epsilon, self._decay_factor})'

    @staticmethod
    def _pick_random_bandit(bandit_list):
        return np.random.choice(bandit_list)

    def choose(self, bandit_list, *args, **kargs):
        epsilon = self._initial_epsilon * (1 - self._decay_factor) ** self._time_steps
        self._time_steps += 1

        if np.random.rand() < epsilon: #explore
            return self._pick_random_bandit(bandit_list), True
        else: #exploit
            return self._pick_best_bandit(bandit_list), False