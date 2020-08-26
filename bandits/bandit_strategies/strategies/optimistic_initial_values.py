from .greedy import Greedy
from ..bandit import Bandit

class OptimisticInitialValues(Greedy):
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __str__(self):
        return f'OptimisticInitialValues({self.initial_value})'

    def init_bandit(self, true_p):
        return Bandit(true_p, initial_p_estimate=self.initial_value)