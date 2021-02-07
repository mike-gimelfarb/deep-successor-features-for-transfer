from collections import defaultdict
import numpy as np
import random


class Q:
    
    def __init__(self, gamma, epsilon, T, epsilon_decay=1., epsilon_min=0., print_ev=1000, save_ev=100, **kwargs):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.T = T
        self.print_ev = print_ev
        self.save_ev = save_ev
    
    def _reset_q(self):
        raise NotImplementedError
    
    def reset(self):
        self.t = 0
        self.cum_reward = 0.
        self.reward_hist, self.episode_reward_hist, self.cum_reward_hist = [], [], []

    def next_task(self, task):
        self.task = task
        self.n_actions = task.action_count()
        self.t += 1
        self.new_episode = True
        self.i = 1
        self.n_eps = 1
        self.eps = self.epsilon
        self.task_reward = 0.;
        self.episode_reward = 0.
        self.last_episode_reward = 0.
        self._reset_q()
    
    def _epsilon_greedy(self, q):
        
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.eps:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)
        
        # decrease the exploration gradually
        self.eps = max(self.eps * self.epsilon_decay, self.epsilon_min)
        return a
    
    def _get_q(self, state):
        raise NotImplementedError
    
    def _update_q(self, state, action, reward, next_state, gamma):
        raise NotImplementedError
        
    def next_sample(self):
        
        # start a new episode
        if self.new_episode:
            self.new_episode = False
            self.s = self.task.initialize()
            self.n_eps += 1
            self.i_since_last_eps = 0
            self.last_episode_reward = self.episode_reward
            self.episode_reward = 0.
        
        # sample from a Bernoulli distribution with parameter epsilon
        q = self._get_q(self.s)
        a = self._epsilon_greedy(q)
        
        # take action a and observe reward r and next state s'
        s1, r, terminal = self.task.transition(a)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma
        
        # update Q for the current task t
        self._update_q(self.s, a, r, s1, gamma)
        
        # monitor performance
        self.task_reward += r
        self.cum_reward += r
        self.episode_reward += r
        if self.i % self.save_ev == 0:
            self.reward_hist.append(self.task_reward)
            self.cum_reward_hist.append(self.cum_reward)
            
        # increment
        self.s = s1
        self.i += 1
        self.i_since_last_eps += 1
        if self.i_since_last_eps >= self.T:
            self.new_episode = True
        
        # printing
        if self.i % self.print_ev == 0:
            basic_str = 't \t {} \t n_eps \t {} \t eps \t {:.5f} \t reward \t {:.5f} \t cum_reward \t {:.5f} \t'.format(
                self.i, self.n_eps, self.eps, self.last_episode_reward, self.task_reward)
            print(basic_str)

        
class TabularQ(Q):
    
    def __init__(self, alpha, *args, noise_init=lambda size: np.random.uniform(low=-0.01, high=0.01, size=size), **kwargs):
        super(TabularQ, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.noise_init = noise_init
    
    def _reset_q(self):
        self.Q = defaultdict(lambda: self.noise_init((self.n_actions,)))
    
    def _get_q(self, state):
        return self.Q[state]
    
    def _update_q(self, state, action, reward, next_state, gamma):
        target = reward + gamma * np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
