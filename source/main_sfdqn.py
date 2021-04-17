# -*- coding: UTF-8 -*-  
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers, Model, optimizers

from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.buffer import ReplayBuffer
from features.deep import DeepSF
from tasks.reacher import Reacher
from utils.config import parse_config_file

# read parameters from config file
config_params = parse_config_file('reacher.cfg')

gen_params = config_params['GENERAL']
n_samples = gen_params['n_samples']

task_params = config_params['TASK']
goals = task_params['train_targets']
test_goals = task_params['test_targets']
all_goals = goals + test_goals
    
agent_params = config_params['AGENT']
dqn_params = config_params['DQN']
sfdqn_params = config_params['SFDQN']


# tasks
def generate_tasks(include_target):
    train_tasks = [Reacher(all_goals, i, include_target) for i in range(len(goals))]
    test_tasks = [Reacher(all_goals, i + len(goals), include_target) for i in range(len(test_goals))]
    return train_tasks, test_tasks


# keras model
def dqn_model_lambda():
    keras_params = dqn_params['keras_params']
    x = y = layers.Input(6)
    for n_neurons, activation in zip(keras_params['n_neurons'], keras_params['activations']):
        y = layers.Dense(n_neurons, activation=activation)(y)
    y = layers.Dense(9, activation='linear')(y)
    model = Model(inputs=x, outputs=y)
    sgd = optimizers.Adam(learning_rate=keras_params['learning_rate'])
    model.compile(sgd, 'mse')
    return model


# keras model for the SF
def sf_model_lambda(x):
    n_features = len(all_goals)
    keras_params = sfdqn_params['keras_params']
    y = x
    for n_neurons, activation in zip(keras_params['n_neurons'], keras_params['activations']):
        y = layers.Dense(n_neurons, activation=activation)(y)
    y = layers.Dense(9 * n_features, activation='linear')(y)
    y = layers.Reshape((9, n_features))(y)
    model = Model(inputs=x, outputs=y)
    sgd = optimizers.Adam(learning_rate=keras_params['learning_rate'])
    model.compile(sgd, 'mse')
    return model


def train():
    
    # build SFDQN    
    print('building SFDQN')
    deep_sf = DeepSF(keras_model_handle=sf_model_lambda, **sfdqn_params)
    sfdqn = SFDQN(deep_sf=deep_sf, buffer=ReplayBuffer(sfdqn_params['buffer_params']),
                  **sfdqn_params, **agent_params)
    
    # train SFDQN
    print('training SFDQN')
    train_tasks, test_tasks = generate_tasks(False)
    sfdqn_perf = sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'])
    
    # build DQN
    print('building DQN')
    dqn = DQN(model_lambda=dqn_model_lambda, buffer=ReplayBuffer(dqn_params['buffer_params']),
              **dqn_params, **agent_params)
    
    # training DQN
    print('training DQN')
    train_tasks, test_tasks = generate_tasks(True)
    dqn_perf = dqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'])

    # smooth data    
    def smooth(y, box_pts):
        return np.convolve(y, np.ones(box_pts) / box_pts, mode='same')

    sfdqn_perf = smooth(sfdqn_perf, 10)[:-5]
    dqn_perf = smooth(dqn_perf, 10)[:-5]
    x = np.linspace(0, 4, sfdqn_perf.size)
    
    # reporting progress
    ticksize = 14
    textsize = 18
    plt.rc('font', size=textsize)  # controls default text sizes
    plt.rc('axes', titlesize=textsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=textsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
    plt.rc('legend', fontsize=ticksize)  # legend fontsize

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(x, sfdqn_perf, label='SFDQN')
    ax.plot(x, dqn_perf, label='DQN')
    plt.xlabel('training task index')
    plt.ylabel('averaged test episode reward')
    plt.title('Testing Reward Averaged over all Test Tasks')
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig('figures/sfdqn_return.png')


train()
