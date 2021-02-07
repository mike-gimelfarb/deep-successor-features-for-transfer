import numpy as np
import random

        
class SFQL:
    """
    Implements the successor feature Q-learning algorithm as introduced in [1]. Suitable for training of
    tasks in sequence in an online manner with different reward functions but shared dynamics.
    
    References
    ----------
    [1] Barreto, André, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.
    """
    
    def __init__(self, sf, gamma, epsilon, T, encoding, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=1000, use_gpi=True, **kwargs):
        """
        Creates a new instance of the successor feature Q-learning algorithm.
        
        Parameters
        ----------
        sf : SF
            an instance of successor features that will be used to train all tasks
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
        use_gpi : boolean
            whether or not to implement policy transfer using GPI according to [1], 
            or to simply train each task separately without transfer
        """
        self.sf = sf
        self.gamma = gamma
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.T = T
        self.encoding = encoding
        self.print_ev = print_ev
        self.save_ev = save_ev
        self.use_gpi = use_gpi
        
    def reset(self):
        """
        Resets the state of the current algorithm and resets the successor feature representation.
        All memory is cleared, including the previous training history.
        """
        
        # reset task information
        self.tasks = []
        
        # reset successor features
        self.sf.reset()
        
        # reset global performance measurements
        self.cum_reward = 0.
        self.reward_hist, self.episode_reward_hist, self.cum_reward_hist = [], [], []
        
    def add_task(self, task, copy_sf_index=-1):
        """
        Adds a successor feature representation for the specified task.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        copy_sf_index : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """
        
        # update task information
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)     
        self.n_actions = task.action_count()
        self.d = task.feature_dim()
        self.phi = task.features
        if self.encoding == 'task':
            self.encoding = task.encode
            
        # update SFs
        self.sf.add_task(task, copy_sf_index)
        
    def set_active_task(self, task_index=None):
        """
        Sets the specified task as the active task.
        
        Parameters
        ----------
        task_index : integer
            the task at the specified index is set to active; if None, this is the most recently-added task
        """
        if task_index is None:
            task_index = self.n_tasks - 1
        self.task_index = task_index
        self.task = self.tasks[task_index]
        
        # reset training memory
        self.s = self.s_enc = None
        self.new_episode = True
        self.steps = self.steps_since_last_episode = self.episode = 0
        self.epsilon = self.epsilon_init
        self.reward = self.reward_since_last_episode = self.episode_reward = 0.
        
    def _epsilon_greedy(self, q):
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
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
            a = np.argmax(q)
        
        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return a
    
    def next_sample(self):
        """
        Queries the next sample from the currently active task. Updates the successor features
        according to the protocol in SFQL in [1] in an online manner, and updates all training history
        periodically.
        """
        
        # start a new episode
        if self.new_episode:
            self.s = self.task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = 0.      
            self.episode_reward_hist.append(self.episode_reward)   

        # generalized policy improvement
        q, c = self.sf.GPI(self.s_enc, self.task_index, update_counters=self.use_gpi)
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
        phi = self.phi(self.s, a, s1)
        self.sf.update_reward(phi, r, self.task_index)
        
        # update SF for the current task t
        q1, _ = self.sf.GPI(s1_enc, self.task_index)
        next_action = np.argmax(np.max(q1, axis=1))
        self.sf.update_successor(self.s_enc, a, phi, s1_enc, next_action, gamma, self.task_index)
        
        # update SF for source task c
        if c != self.task_index:
            q1 = self.sf.GPE(s1_enc, c, c)
            next_action = np.argmax(q1)
            self.sf.update_successor(self.s_enc, a, phi, s1_enc, next_action, gamma, c)
            
        # monitor performance
        self.reward_since_last_episode += r
        self.reward += r
        self.cum_reward += r
        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
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
                self.task_index, self.steps, self.episode, self.epsilon)
            reward_str = 'ep_reward \t {:.5f} \t cum_reward \t {:.5f}'.format(
                self.episode_reward, self.reward)
            gpi_percent = self.sf.GPI_usage_percent(self.task_index)
            w_error = np.linalg.norm(self.sf.fit_w[-1] - self.sf.true_w[-1])
            gpi_str = 'GPI% \t {:.5f} \t w_err \t {:.5f}'.format(gpi_percent, w_error)
            print(sample_str + '\t' + reward_str + '\t' + gpi_str)
            
