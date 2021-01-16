import numpy as np


class SF:
    
    def __init__(self, alpha_w, use_true_reward=False):
        self.alpha_w = alpha_w
        self.use_true_reward = use_true_reward
    
    def build_successor(self, task, source=None):
        raise NotImplementedError
        
    def get_successor(self, state, policy_index):
        raise NotImplementedError  # dimensions: [n_batch, n_actions, n_features]
    
    def get_successors(self, state):
        raise NotImplementedError  # dimensions: [n_batch, n_tasks, n_actions, n_features]
    
    def update_successor(self, state, action, phi, next_state, next_action, gamma, policy_index):
        raise NotImplementedError

    def reset(self):
        self.n_tasks = 0
        self.psi = []
        self.true_w = []
        self.fit_w = []
        
        # statistics
        self.gpi_counters = []

    def add_task(self, task, source=None):
        
        # add successor features to the library
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)
        
        # build new reward function
        true_w = task.get_w()
        self.true_w.append(true_w)
        if self.use_true_reward:
            fit_w = true_w
        else:
            n_features = task.feature_dim()
            fit_w = np.random.uniform(low=-0.01, high=0.01, size=(n_features, 1))
        self.fit_w.append(fit_w)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
        
    def update_reward(self, phi, r, task_index):
        
        # update reward using linear regression
        w = self.fit_w[task_index]
        phi = phi.reshape(w.shape)
        r_fit = np.sum(phi * w)
        self.fit_w[task_index] = w + self.alpha_w * (r - r_fit) * phi
    
        # validate reward
        r_true = np.sum(phi * self.true_w[task_index])
        if r != r_true:
            raise Exception('sampled reward {} != linear reward {} - please check task {}!'.format(r, r_true, task_index))
        
    def GPE(self, state_batch, policy_index, task_index):
        psi = self.get_successor(state_batch, policy_index)
        w = self.fit_w[task_index]
        q = psi @ w  # shape (n_batch, n_actions)
        return q
    
    def GPI(self, state_batch, task_index, update_counters=False):
        psi = self.get_successors(state_batch)
        w = self.fit_w[task_index]
        q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
        task = np.squeeze(np.argmax(np.max(q, axis=2), axis=1))  # shape (n_batch,)
        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task
    
    def GPI_usage_percent(self, task_index):
        counts = self.gpi_counters[task_index]        
        return 1. - (float(counts[task_index]) / np.sum(counts))
