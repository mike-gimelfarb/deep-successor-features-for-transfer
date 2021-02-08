import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras import layers, Model, optimizers 

from successors.deep import DeepSF
from tasks.pinball import Pinball, PinballView
from training.replay import ReplayBuffer
from training.sfdqn import SFDQN

# tasks
goals = [(0.45, 0.3), (0.2, 0.2), (0.75, 0.4), (0.82, 0.08), (0.5, 0.05)]
tasks = [Pinball('tasks/pinball.cfg', goals, i) for i in range(len(goals))]
viewers = [PinballView(task) for task in tasks]

# training params for the SF
sf_params = {
    'alpha_w': 0.5,
    'use_true_reward': True
}

# keras model for the SF
def psi_model_lambda(x):
    actions = 5
    features = len(tasks)
    y = layers.Dense(100, activation='relu')(x)
    y = layers.Dense(100, activation='relu')(y)
    y = layers.Dense(actions * features)(y)
    y = layers.Reshape((actions, features))(y)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizers.Adam(0.001), 'mse')
    return model


# training parameters for experience replay
buffer_params = {
    'n_samples': 200000,
    'n_batch': 32
}

# training params for SFDQN
params = {
    'gamma': 0.96,
    'epsilon': 0.1,
    'T': 600,
    'print_ev': 1000,
    'save_ev': 200,
    'view_ev' : 10,
    'encoding': 'task'
}

# training params for the overall experiment
n_samples = 50000
n_trials = 1

# agents
sfdqn = SFDQN(DeepSF(psi_model_lambda, **sf_params), ReplayBuffer(**buffer_params), **params)
 
# train
cum_data_sfdqn = 0.
for _ in range(n_trials):
    
    # train agent
    sfdqn.train(tasks, n_samples, viewers=viewers)
    
    # update performance statistics
    cum_data_sfdqn = cum_data_sfdqn + np.array(sfdqn.reward_hist) / float(n_trials)

# plot the cumulative return per trial, averaged 
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(cum_data_sfdqn, label='SFDQN')
plt.xlabel('samples')
plt.ylabel('cumulative reward')
plt.legend()
plt.title('Cumulative Training Reward Per Task')
plt.savefig('figures/sfdqn_cumulative_return.png')
plt.show()
