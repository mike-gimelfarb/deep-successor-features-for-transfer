# -*- coding: UTF-8 -*-
from collections import deque
import numpy as np
import random


class ReplayBuffer:
    
    def __init__(self, *args, n_samples=1000000, n_batch=32, **kwargs):
        self.n_samples = n_samples
        self.n_batch = n_batch
    
    def reset(self):
        self.buffer = deque(maxlen=self.n_samples)
    
    def replay(self):
        if len(self.buffer) < self.n_batch: return None
        experiences = list(zip(*random.sample(self.buffer, self.n_batch)))
        states = np.vstack(experiences[0])
        actions = np.array(experiences[1])
        rewards = np.vstack(experiences[2])
        next_states = np.vstack(experiences[3])
        gammas = np.array(experiences[4])
        return states, actions, rewards, next_states, gammas
    
    def append(self, state, action, reward, next_state, gamma):
        self.buffer.append((state, action, reward, next_state, gamma))
    
