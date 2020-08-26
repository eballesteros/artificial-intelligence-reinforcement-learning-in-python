import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from bandit_strategies import Greedy, EpsilonGreedy, DecayingEpsilonGreedy, OptimisticInitialValues

N_TRIALS = int(1e5)
BANDIT_PROBABILITIES = [.2, .5, .75]
            

def run_experiment(strat):
    '''
    :param strat: Strategy; Instance of class implementing strategy API
    '''
    print(strat)

    # init exp recorders
    rewards = []
    explored = []

    # init bandits
    bandits = [strat.init_bandit(p) for p in BANDIT_PROBABILITIES]

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
    cum_rewards = np.cumsum(rewards)
    win_rate = cum_rewards / (np.arange(N_TRIALS) + 1)
    plt.plot(win_rate, label=str(strat))
    plt.plot(np.ones(N_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.ylim(0, 1)
    # plt.show()


if __name__ == '__main__':
    plt.figure()

    run_experiment(EpsilonGreedy(epsilon=.1))
    run_experiment(DecayingEpsilonGreedy(initial_epsilon=.1, decay_factor=.0001))
    run_experiment(OptimisticInitialValues(initial_value=5.))

    plt.legend()
    plt.show()