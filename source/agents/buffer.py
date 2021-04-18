# -*- coding: UTF-8 -*-
import numpy as np


class ReplayBuffer:
    
    def __init__(self, *args, n_samples=1000000, n_batch=32, **kwargs):
        """
        Creates a new randomized replay buffer.
        
        Parameters
        ----------
        n_samples : integer
            the maximum number of samples that can be stored in the buffer
        n_batch : integer
            the batch size
        """
        self.n_samples = n_samples
        self.n_batch = n_batch
    
    def reset(self):
        """
        Removes all samples currently stored in the buffer.
        """
        self.buffer = np.empty(self.n_samples, dtype=object)
        self.index = 0
        self.size = 0
    
    def replay(self):
        """
        Samples a batch of samples from the buffer randomly. If the number of samples
        currently in the buffer is less than the batch size, returns None.
        
        Returns
        -------
        states : np.ndarray
            a collection of starting states of shape [n_batch, -1]
        actions : np.ndarray
            a collection of actions taken in the starting states of shape [n_batch,]
        rewards : np.ndarray:
            a collection of rewards (for DQN) or features (for SFDQN) obtained of shape [n_batch, -1]
        next_states : np.ndarray
            a collection of successor states of shape [n_batch, -1]
        gammas : np.ndarray
            a collection of discount factors to be applied in computing targets for training of shape [n_batch,]
        """
        if self.size < self.n_batch: return None
        indices = np.random.randint(low=0, high=self.size, size=(self.n_batch,))
        states, actions, rewards, next_states, gammas = zip(*self.buffer[indices])
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        gammas = np.array(gammas)
        return states, actions, rewards, next_states, gammas
    
    def append(self, state, action, reward, next_state, gamma):
        """
        Adds the specified sample to the replay buffer. If the buffer is full, then the earliest added
        sample is removed, and the new sample is added.
        
        Parameters
        ----------
        state : np.ndarray
            the encoded state of the task
        action : integer
            the action taken in state
        reward : float or np.ndarray
            the reward obtained in the current transition (for DQN) or state features (for SFDQN)
        next_state : np.ndarray
            the encoded successor state
        gamma : floag
            the effective discount factor to be applied in computing targets for training
        """
        self.buffer[self.index] = (state, action, reward, next_state, gamma)
        self.size = min(self.size + 1, self.n_samples)
        self.index = (self.index + 1) % self.n_samples
        
