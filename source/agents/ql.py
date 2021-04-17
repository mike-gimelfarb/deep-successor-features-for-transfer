# -*- coding: UTF-8 -*-
from collections import defaultdict
import numpy as np

from agents.agent import Agent


class QL(Agent):
    
    def __init__(self, learning_rate, *args, **kwargs):
        """
        Creates a new tabular Q-learning agent.
        
        Parameters
        ----------
        learning_rate : float
            the learning rate to use in order to update Q-values
        """
        super(QL, self).__init__(*args, **kwargs)
        self.alpha = learning_rate
        
    def get_Q_values(self, s, s_enc):
        return self.Q[s]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        target = r + gamma * np.max(self.Q[s1])
        error = target - self.Q[s][a]
        self.Q[s][a] += self.alpha * error
        
    def set_active_training_task(self, index):
        super(QL, self).set_active_training_task(index)
        self.Q = defaultdict(lambda: np.random.uniform(low=-0.01, high=0.01, size=(self.n_actions,)))

