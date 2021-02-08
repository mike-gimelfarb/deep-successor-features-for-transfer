from collections import deque
import numpy as np
import random


class ReplayBuffer:
    """
    A storage container in which to store, and from which to sample, previously-observed
    transitions from an MDP environment. Implements the randomized replay algorithm in [1].
    
    References
    ----------
    [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
    """
    
    def __init__(self, n_samples=1000000, n_batch=32):
        """
        Creates a new instance of a randomized experience replay buffer.
        
        Parameters
        ----------
        n_samples : integer
            the maximum capacity of the replay buffer
        n_batch : integer
            the batch size for sampling experiences
        """
        self.n_samples = n_samples
        self.n_batch = n_batch
    
    def reset(self):
        """
        Clears the current buffer by removing all previously added experiences.
        """
        self.buffer = deque(maxlen=self.n_samples)
    
    def replay(self):
        """
        Samples a batch of experiences using random uniform sampling from the buffer.
        
        Returns
        -------
        np.ndarray : batch of states
        np.ndarray : batch of actions
        np.ndarray : batch of rewards
        np.ndarray : batch of next (successor) states
        np.ndarray : batch of discount factors        
        """
        if len(self.buffer) < self.n_batch: 
            return None
        experiences = list(zip(*random.sample(self.buffer, self.n_batch)))
        states = np.vstack(experiences[0])
        actions = np.array(experiences[1])
        rewards = np.vstack(experiences[2])
        next_states = np.vstack(experiences[3])
        gammas = np.array(experiences[4])
        return states, actions, rewards, next_states, gammas
    
    def append(self, state, action, reward, next_state, gamma):
        """
        Adds the experience to the buffer. If the buffer is currently full, removes the oldest entry first.
        
        Parameters
        ----------
        state : object
            state of the MDP
        action : integer
            action selected in the state
        reward : float
            reward observed in the current transition
        next_state : object
            the next state of the MDP
        gamma : float
            the discount factor
        """
        self.buffer.append((state, action, reward, next_state, gamma))
    
