# -*- coding: UTF-8 -*-
import random
import numpy as np

from agents.agent import Agent


class SFDQN(Agent):
    
    def __init__(self, deep_sf, buffer, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
        """
        Creates a new SFDQN agent per the specifications in the original paper.
        
        Parameters
        ----------
        deep_sf : DeepSF
            instance of deep successor feature representation
         buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(SFDQN, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.buffer = buffer
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon
        
    def get_Q_values(self, s, s_enc):
        q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = self.task_index
        self.c = c
        return q[:, c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # update w
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, self.task_index)
        
        # remember this experience
        self.buffer.append(s_enc, a, phi, s1_enc, gamma)
        
        # update SFs
        transitions = self.buffer.replay()
        for index in range(self.n_tasks):
            self.sf.update_successor(transitions, index)
        
    def reset(self):
        super(SFDQN, self).reset()
        self.sf.reset()
        self.buffer.reset()

    def add_training_task(self, task):
        super(SFDQN, self).add_training_task(task)
        self.sf.add_training_task(task, source=None)
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFDQN, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.5f} \t w_err \t {:.5f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
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
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
                    avg_R = np.mean(Rs)
                    return_data.append(avg_R)
        return return_data
    
    def get_test_action(self, s_enc, w):
        if random.random() <= self.test_epsilon:
            a = random.randrange(self.n_actions)
        else:
            q, c = self.sf.GPI_w(s_enc, w)
            q = q[:, c,:]
            a = np.argmax(q)
        return a
            
    def test_agent(self, task):
        R = 0.0
        w = task.get_w()
        s = task.initialize()
        s_enc = self.encoding(s)
        for _ in range(self.T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            s, s_enc = s1, s1_enc
            R += r
            if done:
                break
        return R
    
