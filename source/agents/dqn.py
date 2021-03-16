# -*- coding: UTF-8 -*-
import numpy as np

from agents.agent import Agent


class DQN(Agent):
    
    def __init__(self, model_lambda, buffer, *args, target_update_ev=1000, **kwargs):
        super(DQN, self).__init__(*args, **kwargs)
        self.model_lambda = model_lambda
        self.buffer = buffer
        self.target_update_ev = target_update_ev
        
    def get_Q_values(self, s, s_enc):
        return self.Q.predict(s_enc)
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        self.buffer.append(s_enc, a, r, s1_enc, gamma)
        
        # sample experience at random
        batch = self.buffer.replay()
        if batch is None: 
            return
        states, actions, rewards, next_states, gammas = batch
        n_batch = self.buffer.n_batch
        indices = np.arange(n_batch)
        rewards = rewards.flatten()

        # main update
        next_actions = np.argmax(self.Q.predict(next_states), axis=1)
        targets = self.Q.predict(states)
        targets[indices, actions] = rewards + gammas * self.target_Q.predict(next_states)[indices, next_actions]
        self.Q.fit(states, targets, verbose=False, batch_size=n_batch)
        
        # target update
        self.updates_since_target_updated += 1
        if self.updates_since_target_updated >= self.target_update_ev:
            self.target_Q.set_weights(self.Q.get_weights())
            self.updates_since_target_updated = 0
    
    def set_active_training_task(self, index):
        super(DQN, self).set_active_training_task(index)
        self.Q = self.model_lambda()
        self.target_Q = self.model_lambda()
        self.target_Q.set_weights(self.Q.get_weights())
        self.buffer.reset()
        self.updates_since_target_updated = 0
            
