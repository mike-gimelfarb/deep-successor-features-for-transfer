# -*- coding: UTF-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt    
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
tf.compat.v1.disable_v2_behavior()

from agents.dqn import DQN
from agents.sfdqn import SFDQN
from agents.buffer import ReplayBuffer
from features.deep import DeepSF
from tasks.reacher import Reacher
from utils.config import parse_config_file

# general training params
config_params = parse_config_file('reacher.cfg')
gen_params = config_params['GENERAL']
task_params = config_params['TASK']
agent_params = config_params['AGENT']
dqn_params = config_params['DQN']
sfdqn_params = config_params['SFDQN']
goals = task_params['train_targets']
test_goals = task_params['test_targets']


# tasks
def generate_tasks():
    all_goals = goals + test_goals
    train_tasks = [Reacher(all_goals, index) for index in range(len(goals))]
    test_tasks = [Reacher(all_goals, index + len(goals)) for index in range(len(test_goals))]
    return train_tasks, test_tasks


# keras model
def dqn_model_lambda():
    keras_params = dqn_params['keras_params']
    x = y = layers.Input(keras_params['n_states'])
    for n_neurons, activation in zip(keras_params['n_neurons'], keras_params['activations']):
        y = layers.Dense(n_neurons,
                         activation=activation)(y)
    y = layers.Dense(keras_params['n_actions'],
                     activation='linear')(y)
    model = Model(inputs=x, outputs=y)
    sgd = optimizers.Adam(learning_rate=keras_params['learning_rate'])
    model.compile(sgd, 'mse')
    return model


# keras model for the SF
def sf_model_lambda(x):
    n_features = len(goals) + len(test_goals)
    keras_params = sfdqn_params['keras_params']
    y = x
    for n_neurons, activation in zip(keras_params['n_neurons'], keras_params['activations']):
        y = layers.Dense(n_neurons,
                         activation=activation)(y)
    y = layers.Dense(keras_params['n_actions'] * n_features,
                     activation='linear')(y)
    y = layers.Reshape((keras_params['n_actions'], n_features))(y)
    model = Model(inputs=x, outputs=y)
    sgd = optimizers.Adam(learning_rate=keras_params['learning_rate'])
    model.compile(sgd, 'mse')
    return model


deep_sf = DeepSF(keras_model_handle=sf_model_lambda, **sfdqn_params)
sfdqn = SFDQN(deep_sf=deep_sf,
              buffer=ReplayBuffer(agent_params['buffer_params']),
              **sfdqn_params, **agent_params)
# dqn = DQN(model_lambda=dqn_model_lambda,
#           buffer=ReplayBuffer(agent_params['buffer_params']),
#           **dqn_params, **agent_params)
agents = [sfdqn]
names = ['SFDQN']

# training params for experiment 
n_samples = gen_params['n_samples']
n_trials = gen_params['n_trials']
for _ in range(n_trials):
    
    # train each agent on a set of tasks
    train_tasks, test_tasks = generate_tasks()
    perfs = []
    for agent, name in zip(agents, names):
        print('\nsolving with {}'.format(name))
        perf = agent.train(train_tasks, n_samples, test_tasks=test_tasks)
        perfs.append(perf)

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
for name, perf in zip(names, perfs):
    plt.plot(perf, label=name)
plt.legend(frameon=False)
plt.xlabel('sample')
plt.ylabel('test reward')
plt.title('Test Reward - Reacher Domain')
plt.tight_layout()
plt.show()
