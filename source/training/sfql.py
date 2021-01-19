import numpy as np
import random

        
class SFQL:
    
    def __init__(self, sf, gamma, epsilon, T, encoding, epsilon_decay=1., epsilon_min=0., print_ev=1000, use_gpi=True):
        self.sf = sf
        self.gamma = gamma
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.T = T
        self.encoding = encoding
        self.print_ev = print_ev
        self.use_gpi = use_gpi
        
    def reset(self):
        self.sf.reset()
        self.cum_reward_hist = []
        
    def next_task(self, task):
        self.sf.add_task(task, -1)
        self.task = task
        self.t = self.sf.n_tasks - 1
        self.n_actions = task.action_count()
        self.d = task.feature_dim()
        self.phi = task.features
        if self.encoding == 'task':
            self.encoding = task.encode
        
        # memory
        self.new_episode = True
        self.steps = self.steps_since_last_episode = self.episode = 0
        self.epsilon = self.epsilon_init
        self.cum_reward = self.reward_since_last_episode = self.episode_reward = 0.
        self.s = self.s_enc = None
        
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
    
    def next_sample(self):
        
        # start a new episode
        if self.new_episode:
            self.s = self.task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = 0.      

        # generalized policy improvement
        q, c = self.sf.GPI(self.s_enc, self.t, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = self.t
        
        # sample from a Bernoulli distribution with parameter epsilon
        a = self._epsilon_greedy(q[:, c,:])
        
        # take action a and observe reward r and next state s'
        s1, r, terminal = self.task.transition(a)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma
        
        # update w
        phi = self.phi(self.s, a, s1)
        self.sf.update_reward(phi, r, self.t)
        
        # update SF for the current task t
        q1, _ = self.sf.GPI(s1_enc, -1)
        next_action = np.argmax(np.max(q1[0,:,:], axis=0))
        self.sf.update_successor(self.s_enc, a, phi, s1_enc, next_action, gamma, self.t)
        
        # update SF for source task c
        if c != self.t:
            q1 = self.sf.GPE(s1_enc, c, c)
            next_action = np.argmax(q1[0,:])
            self.sf.update_successor(self.s_enc, a, phi, s1_enc, next_action, gamma, c)
            
        # monitor performance
        self.reward_since_last_episode += r
        self.cum_reward += r
        self.cum_reward_hist.append(self.cum_reward)
        
        # increment
        self.s = s1
        self.s_enc = s1_enc
        self.steps += 1
        self.steps_since_last_episode += 1
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
        
        # printing
        if self.steps % self.print_ev == 0:
            sample_str = 'task \t {} \t steps \t {} \t episodes \t {} \t eps \t {:.5f}'.format(
                self.t, self.steps, self.episode, self.epsilon)
            reward_str = 'ep_reward \t {:.5f} \t cum_reward \t {:.5f}'.format(
                self.episode_reward, self.cum_reward)
            gpi_percent = self.sf.GPI_usage_percent(self.t)
            w_error = np.linalg.norm(self.sf.fit_w[-1] - self.sf.true_w[-1])
            gpi_str = 'GPI% \t {:.5f} \t w_err \t {:.5f}'.format(gpi_percent, w_error)
            print(sample_str + '\t' + reward_str + '\t' + gpi_str)
            
