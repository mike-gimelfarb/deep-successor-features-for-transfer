import numpy as np

from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import concatenate, Input, Lambda

from successors.successor import SF


class DeepSF(SF):
    """
    A successor feature representation implemented using Keras. Accepts a wide variety of neural networks as
    function approximators.
    """
    
    def __init__(self, keras_model_handle, *args, target_update_ev=1000, **kwargs):
        """
        Creates a new deep representation of successor features.
        
        Parameters
        ----------
        keras_model_handle : function
            a function from an input tensor to a compiled Keras model for successor features
            the Keras model must have outputs reshaped to [None, n_actions, n_features], where
                None corresponds to the batch dimension
                n_actions is the number of actions of the MDP
                n_features is the number of state features to learn SFs
        target_update_ev : integer 
            how often to update the target network, measured by the number of training calls
        """
        super(DeepSF, self).__init__(*args, **kwargs)
        self.keras_model_handle = keras_model_handle
        self.target_update_ev = target_update_ev
    
    def reset(self):
        SF.reset(self)
        self.updates_since_target_updated = []
        
    def build_successor(self, task, source=None):
        
        # input tensor
        if self.n_tasks == 0:
            n_states = task.encode_dim()
            self.inputs = Input(shape=(n_states,))
            self.all_outputs = None
            
        # build new model in keras and copy its weights if needed
        # output shape is assumed to be [n_batch, n_actions, n_features]
        model = self.keras_model_handle(self.inputs)
        if source is not None and self.n_tasks > 0:
            model.set_weights(self.psi[source].get_weights())
        
        # append predictions of all SF networks across tasks to allow fast prediction
        expand_output = Lambda(lambda x: K.expand_dims(x, axis=1))(model.output)
        if self.all_outputs is None:
            self.all_outputs = expand_output
        else:
            self.all_outputs = concatenate([self.all_outputs, expand_output], axis=1)
        self.all_output_model = Model(inputs=self.inputs, outputs=self.all_outputs)
        self.all_output_model.compile('sgd', 'mse')  # dummy compile so Keras doesn't complain
        
        # build target models and copy their weights 
        target_model = self.keras_model_handle(self.inputs)
        target_model.set_weights(model.get_weights())
        self.updates_since_target_updated.append(0)
        
        return model, target_model
        
    def get_successor(self, state, policy_index):
        psi, _ = self.psi[policy_index]
        return psi.predict(state)
    
    def get_successors(self, state):
        return self.all_output_model.predict(state)
    
    def update_successor(self, state, action, phi, next_state, next_action, gamma, policy_index):
        
        # compute target
        psi, target_psi = self.psi[policy_index]
        targets = phi.flatten() + gamma * target_psi.predict(next_state)[0, next_action,:]
        
        # train the SF network
        labels = psi.predict(state)
        labels[0, action,:] = targets
        psi.fit(state, labels, verbose=False)
        
        # update the target SF network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            target_psi.set_weights(psi.get_weights())
            self.updates_since_target_updated[policy_index] = 0

    def update_successor_on_batch(self, states, actions, phis, next_states, gammas, policy_index):
        
        # next actions come from GPI
        q, _ = self.GPI(next_states, policy_index)
        next_actions = np.argmax(np.max(q, axis=1), axis=-1)
         
        # compute targets
        psi, target_psi = self.psi[policy_index]
        indices = np.arange(next_actions.size)
        targets = phis + gammas.reshape((-1, 1)) * target_psi.predict(next_states)[indices, next_actions,:]
        
        # train the SF network
        labels = psi.predict(states)
        labels[indices, actions,:] = targets
        psi.fit(states, labels, verbose=False, batch_size=next_actions.size)
         
        # update the target SF network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            target_psi.set_weights(psi.get_weights())
            self.updates_since_target_updated[policy_index] = 0
        
