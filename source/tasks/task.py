class Task:
    """
    An abstract representation of an MDP with arbitrary state space and finite action space.
    """
    
    def clone(self):
        """
        Creates an identical copy of the current environment, for use in testing.
        
        Returns
        -------
        Task : the copy of the current task
        """
        raise NotImplementedError

    def initialize(self):
        """
        Resets the state of the environment.
        
        Returns
        -------
        object : the initial state of the MDP
        """
        raise NotImplementedError
    
    def action_count(self):
        """
        Returns the number of possible actions in the MDP.
        
        Returns
        -------
        integer : number of possible actions
        """
        raise NotImplementedError
    
    def transition(self, action):
        """
        Applies the specified action in the environment, updating the state of the MDP.
        
        Parameters
        ----------
        action : integer
            the action to apply to the environment
        
        Returns 
        -------
        object : the next state of the MDP
        float : the immediate reward observed in the transition
        boolean : whether or not a terminal state has been reached
        """
        raise NotImplementedError
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        """
        Encodes the state of the MDP according to its canonical encoding.
        
        Parameters
        ----------
        state : object
            the state of the MDP to encode
        
        Returns
        -------
        np.ndarray : the encoding of the state
        """
        raise NotImplementedError
    
    def encode_dim(self):
        """
        Returns the dimension of the canonical state encoding.
        
        Returns
        -------
        integer : the dimension of the canonical state encoding
        """
        raise NotImplementedError
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        """
        Computes the state features for the current environment, used for learning successor
        feature representations. First introduced in [1].
        
        Parameters
        ----------
        state : object
            the state of the MDP
        action : integer
            the action selected in the state
        next_state : object
            the next state (successor state) of the MDP
        
        Returns
        -------
        np.ndarray : the state features of the transition
        
        References
        ----------
        [1] Dayan, Peter. "Improving generalization for temporal difference learning: 
        The successor representation." Neural Computation 5.4 (1993): 613-624.
        """
        raise NotImplementedError
    
    def feature_dim(self):
        """
        Returns the dimension of the state feature representation.
        
        Returns
        -------
        integer : the dimension of the state feature representation
        """
        raise NotImplementedError
    
    def get_w(self):
        """
        Returns a vector of parameters that represents the reward function for the current task.
        Mathematically, given the state features phi(s,a,s') and reward parameters w, the reward function
        is represented as r(s,a,s') = < phi(s,a,s'), w >. 
        
        Returns
        -------
        np.ndarray : a linear parameterization of the reward function of the current MDP
        """
        raise NotImplementedError
    
