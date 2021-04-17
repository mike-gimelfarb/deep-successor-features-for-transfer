# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from agents.sfql import SFQL
from agents.ql import QL
from features.tabular import TabularSF
from tasks.gridworld import Shapes
from utils.config import parse_config_file
from utils.stats import OnlineMeanVariance

# general training params
config_params = parse_config_file('gridworld.cfg')
gen_params = config_params['GENERAL']
task_params = config_params['TASK']
agent_params = config_params['AGENT']
sfql_params = config_params['SFQL']
ql_params = config_params['QL']


# tasks
def generate_task():
    rewards = dict(zip(['1', '2', '3'], list(np.random.uniform(low=-1.0, high=1.0, size=3))))
    return Shapes(maze=np.array(task_params['maze']), shape_rewards=rewards)
 

# agents
sfql = SFQL(TabularSF(**sfql_params), **agent_params) 
ql = QL(**agent_params, **ql_params)
agents = [sfql, ql]
names = ['SFQL', 'QLearning']

# train
data_task_return = [OnlineMeanVariance() for _ in agents]
n_trials = gen_params['n_trials']
n_samples = gen_params['n_samples']
n_tasks = gen_params['n_tasks']
for trial in range(n_trials):
    
    # train each agent on a set of tasks
    for agent in agents:
        agent.reset()
    for t in range(n_tasks):
        task = generate_task()
        for agent, name in zip(agents, names):
            print('\ntrial {}, solving with {}'.format(trial, name))
            agent.train_on_task(task, n_samples)
             
    # update performance statistics 
    for i, agent in enumerate(agents):
        data_task_return[i].update(agent.reward_hist)

# plot the task return
ticksize = 14
textsize = 18
figsize = (20, 10)

plt.rc('font', size=textsize)  # controls default text sizes
plt.rc('axes', titlesize=textsize)  # fontsize of the axes title
plt.rc('axes', labelsize=textsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
plt.rc('legend', fontsize=ticksize)  # legend fontsize

plt.figure(figsize=(12, 6))
ax = plt.gca()
for i, name in enumerate(names):
    mean = data_task_return[i].mean
    n_sample_per_tick = n_samples * n_tasks // mean.size
    x = np.arange(mean.size) * n_sample_per_tick
    se = data_task_return[i].calculate_standard_error()
    plt.plot(x, mean, label=name)
    ax.fill_between(x, mean - se, mean + se, alpha=0.3)
plt.xlabel('sample')
plt.ylabel('cumulative reward')
plt.title('Cumulative Training Reward Per Task')
plt.tight_layout()
plt.legend(ncol=2, frameon=False)
plt.savefig('figures/sfql_return.png')
