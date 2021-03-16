# -*- coding: UTF-8 -*-
import random
import numpy as np


class Agent:
    
    def __init__(self, gamma, T, encoding, *args, epsilon=0.1, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=100, **kwargs):
        self.gamma = gamma
        self.T = T
        if encoding is None:
            encoding = lambda s: s
        self.encoding = encoding
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.print_ev = print_ev
        self.save_ev = save_ev
        if len(args) != 0 or len(kwargs) != 0:
            print(self.__class__.__name__ + ' ignoring parameters ' + str(args) + ' and ' + str(kwargs))
        
    def get_Q_values(self, s, s_enc):
        raise NotImplementedError
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        raise NotImplementedError
    
    # ===========================================================================
    # TASK MANAGEMENT
    # ===========================================================================
    def reset(self):
        self.tasks = []
        self.phis = []
        
        # reset counter history
        self.cum_reward = 0.
        self.reward_hist = []
        self.cum_reward_hist = []
    
    def add_training_task(self, task):
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)  
        self.phis.append(task.features)               
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            if self.encoding == 'task':
                self.encoding = task.encode
    
    def set_active_training_task(self, index):
        
        # set the task
        self.task_index = index
        self.active_task = self.tasks[index]
        self.phi = self.phis[index]
        
        # reset task-dependent counters
        self.s = self.s_enc = None
        self.new_episode = True
        self.episode, self.episode_reward = 0, 0.
        self.steps_since_last_episode, self.reward_since_last_episode = 0, 0.
        self.steps, self.reward = 0, 0.
        self.epsilon = self.epsilon_init
        self.episode_reward_hist = []
        
    # ===========================================================================
    # TRAINING
    # ===========================================================================
    def _epsilon_greedy(self, q):
        assert q.size == self.n_actions
        
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)
        
        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return a
    
    def get_progress_strings(self):
        sample_str = 'task \t {} \t steps \t {} \t episodes \t {} \t eps \t {:.5f}'.format(
            self.task_index, self.steps, self.episode, self.epsilon)
        reward_str = 'ep_reward \t {:.5f} \t reward \t {:.5f}'.format(
            self.episode_reward, self.reward)
        return sample_str, reward_str
    
    def next_sample(self, viewer=None, n_view_ev=None):
        
        # start a new episode
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = 0.   
            if self.episode > 1:
                self.episode_reward_hist.append(self.episode_reward)  
        
        # compute the Q-values in the current state
        q = self.get_Q_values(self.s, self.s_enc)
        
        # choose an action using the epsilon-greedy policy
        a = self._epsilon_greedy(q)
        
        # take action a and observe reward r and next state s'
        s1, r, terminal = self.active_task.transition(a)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma
        
        # train the agent
        self.train_agent(self.s, self.s_enc, a, r, s1, s1_enc, gamma)
        
        # update counters
        self.s, self.s_enc = s1, s1_enc
        self.steps += 1
        self.reward += r
        self.steps_since_last_episode += 1
        self.reward_since_last_episode += r
        self.cum_reward += r
        
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
            
        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
            self.cum_reward_hist.append(self.cum_reward)
        
        # viewing
        if viewer is not None and self.episode % n_view_ev == 0:
            viewer.update()
        
        # printing
        if self.steps % self.print_ev == 0:
            print('\t'.join(self.get_progress_strings()))
    
    def train_on_task(self, train_task, n_samples, viewer=None, n_view_ev=None):
        self.add_training_task(train_task)
        self.set_active_training_task(self.n_tasks - 1)
        for _ in range(n_samples):
            self.next_sample(viewer, n_view_ev)
            
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
        self.reset()
        for train_task, viewer in zip(train_tasks, viewers):
            self.train_on_task(train_task, n_samples, viewer, n_view_ev)
    
