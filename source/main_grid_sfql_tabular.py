import numpy as np

from successors.tabular import TabularSF
from tasks.gridworld import Shapes
from training.sfql import SFQL 
from training.ql import TabularQ 

# task layout
maze = np.array([
    ['1', ' ', ' ', ' ', ' ', '2', 'X', ' ', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '1', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['2', ' ', ' ', ' ', ' ', '3', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['X', 'X', '3', ' ', 'X', 'X', 'X', 'X', 'X', ' ', '1', 'X', 'X'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', '2', ' ', ' ', ' ', ' ', '3'],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', '2', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' '],
    ['_', ' ', ' ', ' ', ' ', ' ', 'X', '3', ' ', ' ', ' ', ' ', '1']
])

# training params for the SF
sf_params = {
    'alpha': 0.5,
    'alpha_w': 0.5,
    'use_true_reward': False
}

# training params for SFQL
params = {
    'gamma': 0.95,
    'epsilon': 0.15,
    'T': 200,
    'print_ev': 1000,
    'encoding': lambda s: s
}

# training params for Q
params_q = {
    'alpha': 0.5
}

# training params for the overall experiment
n_samples = 20000
n_tasks = 20
n_trials = 20

# agents
sfql = SFQL(TabularSF(**sf_params), use_gpi=True, **params)
sf = SFQL(TabularSF(**sf_params), use_gpi=False, **params)
q = TabularQ(**params_q, **params)

# train
avg_data_sfql, cum_data_sfql = 0., 0.
avg_data_sf, cum_data_sf = 0., 0.
avg_data_q, cum_data_q = 0., 0.

for _ in range(n_trials):
    
    # prepare for the next trial
    sfql.reset()
    sf.reset()
    q.reset()
    
    # next trial
    for _ in range(n_tasks):
        
        # define new task
        rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
        task = Shapes(maze, rewards)
        
        # solve the task with sfql
        print('\nsolving with SFQL')
        sfql.next_task(task)
        for _ in range(n_samples):
            sfql.next_sample()
        
        # solve the task with sf
        print('\nsolving with SF')
        sf.next_task(task)
        for _ in range(n_samples):
            sf.next_sample()
        
        # solve the same task with q
        print('\nsolving with QL')
        q.next_task(task)
        for _ in range(n_samples):
            q.next_sample()
    
    # update performance statistics
    avg_data_sfql = avg_data_sfql + np.array(sfql.cum_reward_hist) / float(n_trials)
    cum_data_sfql = cum_data_sfql + np.cumsum(sfql.cum_reward_hist) / float(n_trials)
    avg_data_sf = avg_data_sf + np.array(sf.cum_reward_hist) / float(n_trials)
    cum_data_sf = cum_data_sf + np.cumsum(sf.cum_reward_hist) / float(n_trials)
    avg_data_q = avg_data_q + np.array(q.cum_reward_hist) / float(n_trials)
    cum_data_q = cum_data_q + np.cumsum(q.cum_reward_hist) / float(n_trials)

# plot the cumulative return per trial, averaged 
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(avg_data_sfql, label='SF+GPI')
plt.plot(avg_data_sf, label='SF')
plt.plot(avg_data_q, label='Q')
plt.xlabel('samples')
plt.ylabel('cumulative reward')
plt.legend()
plt.title('Cumulative Training Reward Per Task')
plt.savefig('figures/sfql_tabular_gridworld_cumulative_return_per_task.png')
plt.show()

# plot the gross cumulative return, averaged
plt.clf()
plt.figure(figsize=(5, 5))
plt.plot(cum_data_sfql, label='SF+GPI')
plt.plot(cum_data_sf, label='SF')
plt.plot(cum_data_q, label='Q')
plt.xlabel('samples')
plt.ylabel('cumulative reward')
plt.legend()
plt.title('Total Cumulative Training Reward')
plt.savefig('figures/sfql_tabular_gridworld_cumulative_return_total.png')
plt.show()
