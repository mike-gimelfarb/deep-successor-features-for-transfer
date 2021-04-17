# -*- coding: UTF-8 -*-
import numpy as np
import random

from agents.agent import Agent


class DQN(Agent):
    
    def __init__(self, model_lambda, buffer, *args, target_update_ev=1000, test_epsilon=0.03, **kwargs):
        """
        Creates a new DQN agent that supports universal value function approximation (UVFA).
        
        Parameters
        ----------
        model_lambda : function
            returns a keras Model instance
        buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        target_update_ev : integer
            how often to update the target network (defaults to 1000)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(DQN, self).__init__(*args, **kwargs)
        self.model_lambda = model_lambda
        self.buffer = buffer
        self.target_update_ev = target_update_ev
        self.test_epsilon = test_epsilon
    
    def reset(self):
        Agent.reset(self)
        self.Q = self.model_lambda()
        self.target_Q = self.model_lambda()
        self.target_Q.set_weights(self.Q.get_weights())
        self.buffer.reset()
        self.updates_since_target_updated = 0
        
    def get_Q_values(self, s, s_enc):
        return self.Q.predict_on_batch(s_enc)
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        self.buffer.append(s_enc, a, r, s1_enc, gamma)
        
        # sample experience at random
        batch = self.buffer.replay()
        if batch is None: return
        states, actions, rewards, next_states, gammas = batch
        n_batch = self.buffer.n_batch
        indices = np.arange(n_batch)
        rewards = rewards.flatten()

        # main update
        next_actions = np.argmax(self.Q.predict_on_batch(next_states), axis=1)
        targets = self.Q.predict_on_batch(states)
        targets[indices, actions] = rewards + \
            gammas * self.target_Q.predict_on_batch(next_states)[indices, next_actions]
        self.Q.train_on_batch(states, targets)
        
        # target update
        self.updates_since_target_updated += 1
        if self.updates_since_target_updated >= self.target_update_ev:
            self.target_Q.set_weights(self.Q.get_weights())
            self.updates_since_target_updated = 0
    
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
            
        # train each one
        return_data = []
        for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
            self.set_active_training_task(index)
            for t in range(n_samples):
                
                # train
                self.next_sample(viewer, n_view_ev)
                
                # test
                if t % n_test_ev == 0:
                    Rs = []
                    for test_task in test_tasks:
                        R = self.test_agent(test_task)
                        Rs.append(R)
                    avg_R = np.mean(Rs)
                    return_data.append(avg_R)
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
        return return_data
    
    def get_test_action(self, s_enc):
        if random.random() <= self.test_epsilon:
            a = random.randrange(self.n_actions)
        else:
            q = self.get_Q_values(s_enc, s_enc)
            a = np.argmax(q)
        return a
            
    def test_agent(self, task):
        R = 0.
        s = task.initialize()
        s_enc = self.encoding(s)
        for _ in range(self.T):
            a = self.get_test_action(s_enc)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            s, s_enc = s1, s1_enc
            R += r
            if done:
                break
        return R
