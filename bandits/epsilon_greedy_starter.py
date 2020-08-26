import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from bandit import Bandit
from strategies import EpsilonGreedy, DecayingEpsilonGreedy

N_TRIALS = int(1e5)
BANDIT_PROBABILITIES = [.2, .5, .75]
            

def run_experiment(strategy_class, plot=True, **strat_params):
    # init exp recorders
    rewards = []
    explored = []

    # init bandits
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    # init strat
    strat = strategy_class(**strat_params)
    print(strat)

    # main loop
    for _ in tqdm(range(N_TRIALS)):
        chosen_bandit, exploring = strat.choose(bandits)

        # update recorders
        rewards.append(chosen_bandit.pull())
        explored.append(exploring)

    #report
    print(f'TOTAL REWARD: {sum(rewards)}')
    print(f'WIN RATE: {100 * sum(rewards)/N_TRIALS:.4f}%')
    print(f'TOTAL TIMES EXPLORING: {sum(explored)}')
    print(f'BANDIT ESTIMATES: {[b.p_estimate for b in bandits]}')

    #plot
    if plot:
        cum_rewards = np.cumsum(rewards)
        win_rate = cum_rewards / (np.arange(N_TRIALS) + 1)
        plt.plot(win_rate)
        plt.plot(np.ones(N_TRIALS) * np.max(BANDIT_PROBABILITIES))
        plt.title(str(strat))
        plt.ylim(0, 1)
        plt.show()


if __name__ == '__main__':
    run_experiment(EpsilonGreedy, plot=False, epsilon=.1)
    run_experiment(DecayingEpsilonGreedy, plot=False, initial_epsilon=.1, decay_factor=.0001) 