# -*- coding: UTF-8 -*-
import numpy as np

from agents.agent import Agent


class SFQL(Agent):
    
    def __init__(self, lookup_table, *args, use_gpi=True, **kwargs):
        """
        Creates a new tabular successor feature agent.
        
        Parameters
        ----------
        lookup_table : TabularSF
            a tabular successor feature representation
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        """
        super(SFQL, self).__init__(*args, **kwargs)
        self.sf = lookup_table
        self.use_gpi = use_gpi
        
    def get_Q_values(self, s, s_enc):
        q, self.c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            self.c = self.task_index
        return q[:, self.c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # update w
        t = self.task_index
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, t)
        
        # update SF for the current task t
        if self.use_gpi:
            q1, _ = self.sf.GPI(s1_enc, t)
            q1 = np.max(q1[0,:,:], axis=0)
        else:
            q1 = self.sf.GPE(s1_enc, t, t)[0,:]
        next_action = np.argmax(q1)
        transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
        self.sf.update_successor(transitions, t)
        
        # update SF for source task c
        if self.c != t:
            q1 = self.sf.GPE(s1_enc, self.c, self.c)
            next_action = np.argmax(q1)
            transitions = [(s_enc, a, phi, s1_enc, next_action, gamma)]
            self.sf.update_successor(transitions, self.c)
    
    def reset(self):
        super(SFQL, self).reset()
        self.sf.reset()
        
    def add_training_task(self, task):
        super(SFQL, self).add_training_task(task)
        self.sf.add_training_task(task, -1)
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFQL, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
    
