import numpy as np
import random

        
class SFDQN(): 

    def __init__(self, sf, buffer, gamma, epsilon, T, encoding, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=1000, view_ev=10, use_gpi=True, **kwargs):
        """
        Creates a new instance of the successor feature DQN algorithm.
        
        Parameters
        ----------
        sf : SF
            an instance of successor features that will be used to train all tasks
        buffer : ReplayBuffer
            a replay buffer for storing, retrieving and sharing transitions from the MDP
        gamma : float
            the discount factor
        epsilon : float
            the initial exploration parameter for epsilon-greedy
        T : integer
            the maximum length of each training episode
        encoding : function
            an np.ndarray-value function that transforms each state of the MDP into a form usable by the SF
        epsilon_decay : float
            how much to decay epsilon after each time step
        epsilon_min : float
            the smallest value that epsilon is allowed to take
        print_ev : integer
            how often to print training progress to the console
        save_ev : integer
            how often to save training progress to internal memory
        view_ev : integer
            how often to visualize trained policy using the task's viewer object, in terms of episodes
        use_gpi : boolean
            whether or not to implement policy transfer using GPI according to [1], 
            or to simply train each task separately without transfer
        """
        
        self.sf = sf
        self.buffer = buffer
        self.gamma = gamma
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.T = T
        self.encoding = encoding
        self.print_ev = print_ev
        self.save_ev = save_ev
        self.view_ev = view_ev
        self.use_gpi = use_gpi
    
    def reset(self):
        """
        Resets the state of the current algorithm and resets the successor feature representation.
        All memory is cleared, including the previous training history and buffer.
        """
        
        # reset task information
        self.tasks = []
        
        # reset successor features and replay buffer
        self.buffer.reset()
        self.sf.reset()
                
    def set_tasks(self, tasks, viewers=None):
        """
        Adds successor feature representations for the specified tasks.
        
        Parameters
        ----------
        tasks : iterable of Tasks
            a set of training MDP environments for which to learn successor features
        viewers : iterable of TaskViewer
            a set of viewers corresponding to each training task to visualize policy
        """
        
        # update task information
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.n_actions = tasks[-1].action_count()
        self.d = tasks[-1].feature_dim()
        self.phi = tasks[-1].features
        if self.encoding == 'task':
            self.encoding = tasks[-1].encode
        if viewers is None:
            viewers = [None] * self.n_tasks
        self.viewers = viewers
            
        # update SFs
        for task in tasks:
            self.sf.add_task(task)
        
        # training memory objects
        self.steps = 0
        self.s = [None] * self.n_tasks
        self.s_enc = [None] * self.n_tasks
        self.new_episode = [True] * self.n_tasks
        self.episode = [0] * self.n_tasks
        self.steps_since_last_episode = [0] * self.n_tasks
        self.epsilon = [self.epsilon_init] * self.n_tasks
        
        # history and performance
        self.reward = [0.] * self.n_tasks
        self.reward_since_last_episode = [0.] * self.n_tasks
        self.episode_reward = [0.] * self.n_tasks
        self.reward_hist, self.episode_reward_hist = [], []
    
    def set_active_task(self, task_index=None):
        """
        Sets the specified task as the active training task.
        
        Parameters
        ----------
        task_index : integer
            the task at the specified index is set to active; if None, this is the most recently-added task
        """
        
        if task_index is None:
            task_index = self.n_tasks - 1
        self.task_index = task_index
        self.task = self.tasks[task_index]
                
    def _epsilon_greedy(self, q, decay_epsilon=True):
        """
        Samples an action from the epsilon-greedy policy.
        
        Parameters
        ----------
        q : array-like:
            the Q-values used for greedy action selection
        
        Returns
        -------
        integer : the action sampled from the epsilon-greedy policy
        """
        
        assert q.size == self.n_actions
        
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon[self.task_index]:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)
        
        # decrease the exploration gradually
        if decay_epsilon:
            self.epsilon[self.task_index] = max(
                self.epsilon[self.task_index] * self.epsilon_decay, self.epsilon_min)
        
        return a
            
    def next_sample(self):
        """
        Queries the next sample from the currently active task. Updates all successor features
        according to the protocol in SFDQN in [1] in an offline off-policy manner, and updates all 
        training history periodically.
        """
        
        # start a new episode
        if self.new_episode[self.task_index]:
            self.s[self.task_index] = self.task.initialize()
            self.s_enc[self.task_index] = self.encoding(self.s[self.task_index])
            self.new_episode[self.task_index] = False
            self.episode[self.task_index] += 1
            self.steps_since_last_episode[self.task_index] = 0
            self.episode_reward[self.task_index] = self.reward_since_last_episode[self.task_index]
            self.reward_since_last_episode[self.task_index] = 0.      
            self.episode_reward_hist.append(self.episode_reward[:])   
        
        # generalized policy improvement
        s = self.s[self.task_index]
        s_enc = self.s_enc[self.task_index]
        q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = self.task_index
        
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
        phi = self.phi(s, a, s1)
        self.sf.update_reward(phi, r, self.task_index)
        
        # store the experience in the replay buffer
        self.buffer.append(s_enc, a, phi, s1_enc, gamma)
        
        # update SF for train tasks
        for i in range(self.n_tasks): 
            batch = self.buffer.replay()
            if batch is not None:
                self.sf.update_successor_on_batch(*batch, i)
        
        # monitor performance
        self.reward[self.task_index] += r
        self.reward_since_last_episode[self.task_index] += r
        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward[:])
        
        # viewing
        if self.episode[self.task_index] % self.view_ev == 0 and self.viewers[self.task_index] is not None:
            self.viewers[self.task_index].update()
        
        # increment
        self.s[self.task_index] = s1
        self.s_enc[self.task_index] = s1_enc
        self.steps += 1
        self.steps_since_last_episode[self.task_index] += 1
        if self.steps_since_last_episode[self.task_index] >= self.T:
            self.new_episode[self.task_index] = True

        # printing
        if self.steps % self.print_ev == 0:
            sample_str = 'task \t {} \t steps \t {} \t episodes \t {} \t eps \t {:.5f}'.format(
                self.task_index, self.steps, self.episode[self.task_index], self.epsilon[self.task_index])
            reward_str = 'ep_reward \t {:.5f} \t cum_reward \t {:.5f}'.format(
                self.episode_reward[self.task_index], self.reward[self.task_index])
            gpi_percent = self.sf.GPI_usage_percent(self.task_index)
            w_error = np.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
            gpi_str = 'GPI% \t {:.5f} \t w_err \t {:.5f}'.format(gpi_percent, w_error)
            print(sample_str + '\t' + reward_str + '\t' + gpi_str)
    
    def train(self, tasks, n_samples, viewers=None):
        """
        Executes the entire training pipeline for a fixed set of tasks.
        
        Parameters
        ----------
        tasks : iterable of Task
            a sequence of tasks to learn successor features for
        n_samples : integer
            the number of samples to collect from each task during training
        """        
        self.reset()
        self.set_tasks(tasks, viewers=viewers)
        for t in range(self.n_tasks):
            self.set_active_task(t)
            for _ in range(n_samples):
                self.next_sample()
                
