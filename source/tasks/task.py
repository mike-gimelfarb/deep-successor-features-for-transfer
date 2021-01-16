class Task:
    
    def clone(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError
    
    def action_count(self):
        raise NotImplementedError
    
    def transition(self, action):
        raise NotImplementedError
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        raise NotImplementedError
    
    def encode_dim(self):
        raise NotImplementedError
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        raise NotImplementedError
    
    def feature_dim(self):
        raise NotImplementedError
    
    def get_w(self):
        raise NotImplementedError
    
