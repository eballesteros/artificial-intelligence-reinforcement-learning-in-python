import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from bandit import Bandit
from strategies import Greedy, EpsilonGreedy, DecayingEpsilonGreedy

N_TRIALS = int(1e5)
BANDIT_PROBABILITIES = [.2, .5, .75]
            

def run_experiment(strategy_class, bandit_initial_estimates=None, plot=True, **strat_params):
    # init exp recorders
    rewards = []
    explored = []

    # init bandits
    bandit_initial_estimates = bandit_initial_estimates if bandit_initial_estimates else np.zeros(len(BANDIT_PROBABILITIES))
    bandits = [Bandit(p, p0) for p, p0 in zip(BANDIT_PROBABILITIES, bandit_initial_estimates)]

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
        plt.plot(win_rate, label=str(strat))
        plt.plot(np.ones(N_TRIALS) * np.max(BANDIT_PROBABILITIES))
        plt.xscale('log')
        plt.ylim(0, 1)
        # plt.show()


if __name__ == '__main__':
    plt.figure()

    run_experiment(EpsilonGreedy, epsilon=.1)
    run_experiment(DecayingEpsilonGreedy, initial_epsilon=.1, decay_factor=.0001)
    run_experiment(Greedy, bandit_initial_estimates=[5., 5., 5.]) #optimistic initial values

    plt.legend()
    plt.show()