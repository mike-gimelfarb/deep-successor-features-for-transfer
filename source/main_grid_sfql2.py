import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, backend as K

tf.compat.v1.disable_eager_execution()

from rbf import RBFLayer
from successors.deep import DeepSF
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
    'alpha_w': 0.5,
    'use_true_reward': False
}

# training params for SFQL
params = {
    'gamma': 0.95,
    'epsilon': 0.15,
    'epsilon_decay': 1.,
    'epsilon_min': 0.15,
    'T': 200,
    'print_ev': 1000,
    'encoding': 'task'
}

# training params for experiment
n_samples = 20000
n_tasks = 20
n_trials = 10


# keras model for the SF
def model_lambda(x):
    y = RBFLayer(100, 0.5)(x)
    y = layers.Dense(4 * 4)(y)
    y = layers.Reshape((4, 4))(y)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizers.Adam(0.001), 'mse')
    return model


# build agents
sf = DeepSF(keras_model_handle=model_lambda, **sf_params)
sfql = SFQL(sf, **params)
q = TabularQ(alpha=0.5, **params)

# train
avg_data_sfql, cum_data_sfql = 0., 0.
avg_data_q, cum_data_q = 0., 0.

for _ in range(n_trials):
    
    # prepare for next trial
    sfql.reset()
    q.reset()
    K.clear_session()
       
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
        
        # solve the same task with q
        print('\nsolving with QL')
        q.next_task(task)
        for _ in range(n_samples):
            q.next_sample()
    
    # update performance statistics
    avg_data_sfql = avg_data_sfql + np.array(sfql.cum_reward_hist) / float(n_trials)
    cum_data_sfql = cum_data_sfql + np.cumsum(sfql.cum_reward_hist) / float(n_trials)
    avg_data_q = avg_data_q + np.array(q.cum_reward_hist) / float(n_trials)
    cum_data_q = cum_data_q + np.cumsum(q.cum_reward_hist) / float(n_trials)

# plot the cumulative return per trial, averaged 
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(avg_data_sfql, label='sfql')
plt.plot(avg_data_q, label='q')
plt.legend()
plt.title('Cumulative Training Reward Per Task')
plt.savefig('figures/deep_gridworld_cumulative_return_per_task.png')
plt.show()

# plot the gross cumulative return, averaged
plt.clf()
plt.figure(figsize=(5, 5))
plt.plot(cum_data_sfql, label='sfql')
plt.plot(cum_data_q, label='q')
plt.legend()
plt.title('Total Cumulative Training Reward')
plt.savefig('figures/deep_gridworld_cumulative_return_total.png')
plt.show()
