import numpy as np

from tensorflow.keras import backend as K, Model
from tensorflow.keras.layers import concatenate, Input, Lambda

from successors.successor import SF


class DeepSF(SF):
    
    def __init__(self, keras_model_handle, *args, **kwargs):
        super(DeepSF, self).__init__(*args, **kwargs)
        self.keras_model_handle = keras_model_handle
    
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
        
        # append model predictions
        expand_output = Lambda(lambda x: K.expand_dims(x, axis=1))(model.output)
        if self.all_outputs is None:
            self.all_outputs = expand_output
        else:
            self.all_outputs = concatenate([self.all_outputs, expand_output], axis=1)
        self.all_output_model = Model(inputs=self.inputs, outputs=self.all_outputs)
        self.all_output_model.compile('adam', 'mse')
        
        return model    
        
    def get_successor(self, state, policy_index):
        return self.psi[policy_index].predict(state)
    
    def get_successors(self, state):
        return self.all_output_model.predict(state)
    
    def update_successor(self, state, action, phi, next_state, next_action, gamma, policy_index):
        states = np.concatenate([state, next_state], axis=0)
        psi = self.psi[policy_index].predict(states)
        psi[0, action,:] = phi.reshape((-1,)) + gamma * psi[1, next_action,:]
        self.psi[policy_index].fit(state, psi[0:1,:,:], verbose=False)
