import numpy as np

from successors.tabular import TabularSF
from tasks.gridworld import Shapes
from training.sfql import SFQL 
from training.ql import TabularQ 

maze = np.array([
    ['1', ' ', ' ', ' ', ' ', '2', 'X', '2', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', '1', ' ', ' ', ' ', ' ', '3'],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']
])

# agents
sf_params = {
    'alpha': 0.3,
    'alpha_w': 0.5,
    'use_true_reward': False
}

params = {
    'gamma': 0.95,
    'epsilon': 0.15,
    'epsilon_decay': 1.0,
    'epsilon_min': 0.15,
    'T': 200,
    'print_ev': 1000,
    'encoding': lambda s: s
}

sfql = SFQL(TabularSF(**sf_params), **params)
q = TabularQ(alpha=0.5, **params)

# main loop
avg_data_sfql, cum_data_sfql = 0., 0.
avg_data_q, cum_data_q = 0., 0.
trials = 10

for _ in range(trials):
    
    # next trial
    sfql.reset()
    q.reset()
    
    for _ in range(20):
        
        # define new task
        rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
        task = Shapes(maze, rewards)
        
        # solve the task with sfql
        print('\nsolving with SFQL')
        sfql.next_task(task)
        for _ in range(20000):
            sfql.next_sample()
        
        # solve the same task with q
        print('\nsolving with QL')
        q.next_task(task)
        for _ in range(20000):
            q.next_sample()
    
    avg_data_sfql = avg_data_sfql + np.array(sfql.cum_reward_hist) / float(trials)
    cum_data_sfql = cum_data_sfql + np.cumsum(sfql.cum_reward_hist) / float(trials)
    avg_data_q = avg_data_q + np.array(q.cum_reward_hist) / float(trials)
    cum_data_q = cum_data_q + np.cumsum(q.cum_reward_hist) / float(trials)

import matplotlib.pyplot as plt
plt.plot(avg_data_sfql, label='sfql')
plt.plot(avg_data_q, label='q')
plt.legend()
plt.title('Per-Trial Cumulative Training Return')
plt.show()

plt.clf()
plt.plot(cum_data_sfql, label='sfql')
plt.plot(cum_data_q, label='q')
plt.legend()
plt.title('Total Cumulative Training Return')
plt.show()
